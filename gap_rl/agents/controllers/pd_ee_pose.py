from dataclasses import dataclass
from typing import Sequence, Union

import numpy as np
from collections import deque
import sapien.core as sapien
from gap_rl.utils.common import clip_and_scale_action
from gap_rl.utils.sapien_utils import get_entity_by_name, vectorize_pose
from gym import spaces
from scipy.spatial.transform import Rotation, Slerp, RotationSpline

from ..base_controller import BaseController, ControllerConfig
from .pd_joint_pos import PDJointPosController


class PDEEPosController(PDJointPosController):
    config: "PDEEPosControllerConfig"

    def _initialize_joints(self):
        super()._initialize_joints()

        # Pinocchio model to compute IK
        self.pmodel = self.articulation.create_pinocchio_model()
        self.qmask = np.zeros(self.articulation.dof, dtype=bool)
        self.qmask[self.joint_indices] = 1

        if self.config.ee_link:
            self.ee_link = get_entity_by_name(
                self.articulation.get_links(), self.config.ee_link
            )
        else:
            # The child link of last joint is assumed to be the end-effector.
            self.ee_link = self.joints[-1].get_child_link()
        self.ee_link_idx = self.articulation.get_links().index(self.ee_link)

    def _initialize_action_space(self):
        low = np.float32(np.broadcast_to(self.config.lower, 3))
        high = np.float32(np.broadcast_to(self.config.upper, 3))
        self.action_space = spaces.Box(low, high, dtype=np.float32)

    @property
    def ee_pos(self):
        return self.ee_link.pose.p

    @property
    def ee_pose(self):
        return self.ee_link.pose

    @property
    def ee_pose_at_base(self):
        to_base = self.articulation.pose.inv()
        return to_base.transform(self.ee_pose)

    def reset(self):
        super().reset()
        self._target_pose = self.ee_pose_at_base

    def compute_ik(self, target_pose, max_iterations=100):
        # Assume the target pose is defined in the base frame
        result, success, error = self.pmodel.compute_inverse_kinematics(
            self.ee_link_idx,
            target_pose,
            initial_qpos=self.articulation.get_qpos(),
            active_qmask=self.qmask,
            max_iterations=max_iterations,
        )
        if success:
            return result[self.joint_indices]
        else:
            return None

    def compute_target_pose(self, prev_ee_pose_at_base, action):
        # Keep the current rotation and change the position
        if self.config.use_delta:
            delta_pose = sapien.Pose(action)

            if self.config.frame == "base":
                target_pose = delta_pose * prev_ee_pose_at_base
            elif self.config.frame == "ee":
                target_pose = prev_ee_pose_at_base * delta_pose
            else:
                raise NotImplementedError(self.config.frame)
        else:
            assert self.config.frame == "base", self.config.frame
            target_pose = sapien.Pose(action)

        return target_pose

    def set_action(self, action: np.ndarray):
        action = self._preprocess_action(action)

        self._step = 0
        self._start_qpos = self.qpos

        if self.config.use_target:
            prev_ee_pose_at_base = self._target_pose
        else:
            prev_ee_pose_at_base = self.ee_pose_at_base

        self._target_pose = self.compute_target_pose(prev_ee_pose_at_base, action)
        self._target_qpos = self.compute_ik(self._target_pose)
        if self._target_qpos is None:
            self._target_qpos = self._start_qpos

        if self.config.interpolate:
            self._step_size = (self._target_qpos - self._start_qpos) / self._sim_steps
        else:
            self.set_drive_targets(self._target_qpos)

    def get_state(self) -> dict:
        if self.config.use_target:
            return {"target_pose": vectorize_pose(self._target_pose)}
        return {}

    def set_state(self, state: dict):
        if self.config.use_target:
            target_pose = state["target_pose"]
            self._target_pose = sapien.Pose(target_pose[:3], target_pose[3:])


@dataclass
class PDEEPosControllerConfig(ControllerConfig):
    lower: Union[float, Sequence[float]]
    upper: Union[float, Sequence[float]]
    stiffness: Union[float, Sequence[float]]
    damping: Union[float, Sequence[float]]
    force_limit: Union[float, Sequence[float]] = 1e10
    friction: Union[float, Sequence[float]] = 0.0
    ee_link: str = None
    frame: str = "ee"  # [base, ee]
    use_delta: bool = True
    use_target: bool = False
    interpolate: bool = False
    normalize_action: bool = True
    controller_cls = PDEEPosController


class PDEEPoseController(PDEEPosController):
    config: "PDEEPoseControllerConfig"

    def __init__(
        self,
        config: "ControllerConfig",
        articulation: sapien.Articulation,
        control_freq: int,
        sim_freq: int = None,
    ):
        self._cache_size = config.cache_size
        self._tcp_cache = deque(maxlen=self._cache_size)
        self._action_cache = deque(maxlen=self._cache_size)
        super().__init__(config, articulation, control_freq, sim_freq)

    def _initialize_action_space(self):
        low = np.float32(
            np.hstack(
                [
                    np.broadcast_to(self.config.pos_lower, 3),
                    np.broadcast_to(-self.config.rot_bound, 3),
                ]
            )
        )
        high = np.float32(
            np.hstack(
                [
                    np.broadcast_to(self.config.pos_upper, 3),
                    np.broadcast_to(self.config.rot_bound, 3),
                ]
            )
        )
        self.action_space = spaces.Box(low, high, dtype=np.float32)

    def reset(self):
        super().reset()
        self._tcp_cache.clear()
        self._action_cache.clear()

    def _clip_and_scale_action(self, action):
        # NOTE(xiqiang): rotation should be clipped by norm.
        pos_action = clip_and_scale_action(
            action[:3], self._action_space.low[:3], self._action_space.high[:3]
        )
        rot_action = action[3:]
        rot_norm = np.linalg.norm(rot_action)
        if rot_norm > 1:
            rot_action = rot_action / rot_norm
        rot_action = rot_action * self.config.rot_bound
        return np.hstack([pos_action, rot_action])

    def compute_target_pose(self, prev_ee_pose_at_base, action):
        if self.config.smooth:
            self._tcp_cache.append(prev_ee_pose_at_base)
            self._action_cache.append(action)
            cache_size = len(self._action_cache)
            smooth_pos = 0
            rot_mat = np.eye(3)
            for ind in range(cache_size):
                smooth_pos += self._action_cache[ind][:3]
                delta_rot = self._action_cache[ind][3:6]
                rot_mat = rot_mat @ Rotation.from_rotvec(delta_rot).as_matrix()
            smooth_pos /= cache_size
            # smooth_rot = Rotation.from_matrix(rot_mat).as_rotvec() / cache_size
            slerp = Slerp([0, 1], Rotation.from_matrix([np.eye(3), rot_mat]))
            smooth_rot = slerp(np.linspace(0, 1, cache_size+1)).as_rotvec()[1]

        if self.config.use_delta:
            if self.config.smooth:
                delta_pos, delta_rot = smooth_pos, smooth_rot
            else:
                delta_pos, delta_rot = action[0:3], action[3:6]
            delta_quat = Rotation.from_rotvec(delta_rot).as_quat()[[3, 0, 1, 2]]
            delta_pose = sapien.Pose(delta_pos, delta_quat)

            if self.config.frame == "base":
                target_pose = delta_pose * prev_ee_pose_at_base
            elif self.config.frame == "ee":
                target_pose = prev_ee_pose_at_base * delta_pose
            elif self.config.frame == "ee_align":
                # origin at ee but base rotation
                target_pose = delta_pose * prev_ee_pose_at_base
                target_pose.set_p(prev_ee_pose_at_base.p + delta_pos)
            else:
                raise NotImplementedError(self.config.frame)
        else:
            assert self.config.frame == "base", self.config.frame
            target_pos, target_rot = action[0:3], action[3:6]
            target_quat = Rotation.from_rotvec(target_rot).as_quat()[[3, 0, 1, 2]]
            target_pose = sapien.Pose(target_pos, target_quat)

        return target_pose


@dataclass
class PDEEPoseControllerConfig(ControllerConfig):
    pos_lower: Union[float, Sequence[float]]
    pos_upper: Union[float, Sequence[float]]
    rot_bound: float
    stiffness: Union[float, Sequence[float]]
    damping: Union[float, Sequence[float]]
    force_limit: Union[float, Sequence[float]] = 1e10
    friction: Union[float, Sequence[float]] = 0.0
    cache_size: int = 3
    ee_link: str = None
    frame: str = "ee"  # [base, ee, ee_align]
    smooth: bool = False
    use_delta: bool = True
    use_target: bool = False
    interpolate: bool = False
    normalize_action: bool = True
    controller_cls = PDEEPoseController


class PDEEPoseEulerController(PDEEPoseController):
    config: "PDEEPoseEulerControllerConfig"

    def _initialize_action_space(self):
        low = np.float32(
            np.hstack(
                [
                    np.broadcast_to(self.config.pos_lower, 3),
                    np.broadcast_to(-self.config.rot_bound, 3),
                ]
            )
        )
        high = np.float32(
            np.hstack(
                [
                    np.broadcast_to(self.config.pos_upper, 3),
                    np.broadcast_to(self.config.rot_bound, 3),
                ]
            )
        )
        self.action_space = spaces.Box(low, high, dtype=np.float32)

    def _clip_and_scale_action(self, action):
        # NOTE(xiqiang): rotation should be clipped by norm.
        pos_action = clip_and_scale_action(
            action[:3], self._action_space.low[:3], self._action_space.high[:3]
        )
        rot_action = clip_and_scale_action(
            action[3:], self._action_space.low[3:], self._action_space.high[3:]
        )
        return np.hstack([pos_action, rot_action])

    def compute_target_pose(self, prev_ee_pose_at_base, action):
        if self.config.use_delta:
            delta_pos, delta_rot = action[0:3], action[3:6]
            delta_quat = Rotation.from_euler('XYZ', delta_rot).as_quat()[[3, 0, 1, 2]]
            delta_pose = sapien.Pose(delta_pos, delta_quat)

            if self.config.frame == "base":
                target_pose = delta_pose * prev_ee_pose_at_base
            elif self.config.frame == "ee":
                target_pose = prev_ee_pose_at_base * delta_pose
            elif self.config.frame == "ee_align":
                # origin at ee but base rotation
                target_pose = delta_pose * prev_ee_pose_at_base
                target_pose.set_p(prev_ee_pose_at_base.p + delta_pos)
            else:
                raise NotImplementedError(self.config.frame)
        else:
            assert self.config.frame == "base", self.config.frame
            target_pos, target_rot = action[0:3], action[3:6]
            target_quat = Rotation.from_euler('XYZ', target_rot).as_quat()[[3, 0, 1, 2]]
            # target_quat = Rotation.from_rotvec(target_rot).as_quat()[[3, 0, 1, 2]]
            target_pose = sapien.Pose(target_pos, target_quat)

        return target_pose


@dataclass
class PDEEPoseEulerControllerConfig(ControllerConfig):
    pos_lower: Union[float, Sequence[float]]
    pos_upper: Union[float, Sequence[float]]
    rot_bound: Union[float, Sequence[float]]
    stiffness: Union[float, Sequence[float]]
    damping: Union[float, Sequence[float]]
    force_limit: Union[float, Sequence[float]] = 1e10
    friction: Union[float, Sequence[float]] = 0.0
    cache_size: int = 3
    ee_link: str = None
    frame: str = "ee"  # [base, ee, ee_align]
    smooth: bool = False
    use_delta: bool = True
    use_target: bool = False
    interpolate: bool = False
    normalize_action: bool = True
    controller_cls = PDEEPoseEulerController


class PDEEPoseControllerMultiAct(PDEEPoseController):
    config: "PDEEPoseControllerMultiActConfig"

    def __init__(
        self,
        config: "ControllerConfig",
        articulation: sapien.Articulation,
        control_freq: int,
        sim_freq: int = None,
    ):
        super().__init__(config, articulation, control_freq, sim_freq)

    def _initialize_action_space(self):
        low = np.float32(
            np.hstack(
                [
                    np.broadcast_to(self.config.pos_lower, 3 * self._cache_size),
                    np.broadcast_to(-self.config.rot_bound, 3 * self._cache_size),
                ]
            )
        )
        high = np.float32(
            np.hstack(
                [
                    np.broadcast_to(self.config.pos_upper, 3 * self._cache_size),
                    np.broadcast_to(self.config.rot_bound, 3 * self._cache_size),
                ]
            )
        )
        self.action_space = spaces.Box(low, high, dtype=np.float32)

    def _clip_and_scale_action(self, action):
        # NOTE(xiqiang): rotation should be clipped by norm.
        pos_steps = 3 * self._cache_size
        pos_action = clip_and_scale_action(
            action[:pos_steps], self._action_space.low[:pos_steps], self._action_space.high[:pos_steps]
        )
        rot_action = []
        for step in range(self._cache_size):
            rot_action_step = action[pos_steps:pos_steps + 3]
            rot_norm_step = np.linalg.norm(rot_action_step)
            if rot_norm_step > 1:
                rot_action_step = rot_norm_step / rot_norm_step
            rot_action_step = rot_action_step * self.config.rot_bound
            rot_action.append(rot_action_step)
            pos_steps += 3
        return np.hstack([pos_action, rot_action])

    def compute_target_pose(self, prev_ee_pose_at_base, action):
        if self.config.smooth:
            posrot_seq = action.reshape(2, self._cache_size, 3)
            self._action_cache.append(posrot_seq)
            smooth_pos = 0
            rot_mat = np.eye(3)
            cache_size = len(self._action_cache)
            pred_rotvec_list = []
            for ind in range(-cache_size, 0):
                smooth_pos += self._action_cache[ind][0][-ind-1]
                delta_rot = self._action_cache[ind][1][-ind-1]
                pred_rotvec_list.append(delta_rot)
                rot_mat = rot_mat @ Rotation.from_rotvec(delta_rot).as_matrix()
            smooth_pos /= cache_size
            # smooth_rot = Rotation.from_matrix(rot_mat).as_rotvec() / self._cache_size
            # slerp = Slerp([0, 1], Rotation.from_matrix([np.eye(3), rot_mat]))
            # smooth_rot = slerp(np.linspace(0, 1, cache_size + 1)).as_rotvec()[1]
            ## using NLerp
            # mean_quat = Rotation.from_rotvec(pred_rotvec_list).as_quat().mean(0)
            # quats_norm = np.linalg.norm(mean_quat) + 1e-6
            # smooth_rot = Rotation.from_quat(mean_quat / quats_norm).as_rotvec()
            ## using chordal L2 mean
            smooth_rot = Rotation.from_rotvec(pred_rotvec_list).mean().as_rotvec()

        if self.config.use_delta:
            if self.config.smooth:
                delta_pos, delta_rot = smooth_pos, smooth_rot
            else:
                delta_pos, delta_rot = action[0:3], action[3:6]
            delta_quat = Rotation.from_rotvec(delta_rot).as_quat()[[3, 0, 1, 2]]
            delta_pose = sapien.Pose(delta_pos, delta_quat)

            if self.config.frame == "base":
                target_pose = delta_pose * prev_ee_pose_at_base
            elif self.config.frame == "ee":
                target_pose = prev_ee_pose_at_base * delta_pose
            elif self.config.frame == "ee_align":
                # origin at ee but base rotation
                target_pose = delta_pose * prev_ee_pose_at_base
                target_pose.set_p(prev_ee_pose_at_base.p + delta_pos)
            else:
                raise NotImplementedError(self.config.frame)
        else:
            assert self.config.frame == "base", self.config.frame
            target_pos, target_rot = action[0:3], action[3:6]
            target_quat = Rotation.from_rotvec(target_rot).as_quat()[[3, 0, 1, 2]]
            target_pose = sapien.Pose(target_pos, target_quat)

        return target_pose


@dataclass
class PDEEPoseControllerMultiActConfig(ControllerConfig):
    pos_lower: Union[float, Sequence[float]]
    pos_upper: Union[float, Sequence[float]]
    rot_bound: float
    stiffness: Union[float, Sequence[float]]
    damping: Union[float, Sequence[float]]
    force_limit: Union[float, Sequence[float]] = 1e10
    friction: Union[float, Sequence[float]] = 0.0
    cache_size: int = 3
    ee_link: str = None
    frame: str = "ee"  # [base, ee, ee_align]
    smooth: bool = False
    use_delta: bool = True
    use_target: bool = False
    interpolate: bool = False
    normalize_action: bool = True
    controller_cls = PDEEPoseControllerMultiAct


class PDEEPoseEulerControllerMultiAct(PDEEPoseEulerController):
    config: "PDEEPoseEulerControllerMultiActConfig"

    def __init__(
        self,
        config: "ControllerConfig",
        articulation: sapien.Articulation,
        control_freq: int,
        sim_freq: int = None,
    ):
        super().__init__(config, articulation, control_freq, sim_freq)

    def _initialize_action_space(self):
        low = np.float32(
            np.hstack(
                [
                    np.broadcast_to(self.config.pos_lower, 3 * self._cache_size),
                    np.broadcast_to(-self.config.rot_bound, 3 * self._cache_size),
                ]
            )
        )
        high = np.float32(
            np.hstack(
                [
                    np.broadcast_to(self.config.pos_upper, 3 * self._cache_size),
                    np.broadcast_to(self.config.rot_bound, 3 * self._cache_size),
                ]
            )
        )
        self.action_space = spaces.Box(low, high, dtype=np.float32)

    def _clip_and_scale_action(self, action):
        # NOTE(xiqiang): rotation should be clipped by norm.
        pos_steps = 3 * self._cache_size
        pos_action = clip_and_scale_action(
            action[:pos_steps], self._action_space.low[:pos_steps], self._action_space.high[:pos_steps]
        )
        rot_action = clip_and_scale_action(
            action[pos_steps:], self._action_space.low[pos_steps:], self._action_space.high[pos_steps:]
        )
        return np.hstack([pos_action, rot_action])

    def compute_target_pose(self, prev_ee_pose_at_base, action):
        if self.config.smooth:
            poseuler_seq = action.reshape(2, self._cache_size, 3)
            self._action_cache.append(poseuler_seq)
            smooth_pos = 0
            rot_mat = np.eye(3)
            cache_size = len(self._action_cache)
            pred_euler_list = []
            for ind in range(-cache_size, 0):
                smooth_pos += self._action_cache[ind][0][-ind-1]
                delta_euler = self._action_cache[ind][1][-ind-1]
                pred_euler_list.append(delta_euler)
                rot_mat = rot_mat @ Rotation.from_euler('XYZ', delta_euler).as_matrix()
            smooth_pos /= cache_size
            ## using Slerp
            slerp = Slerp([0, 1], Rotation.from_matrix([np.eye(3), rot_mat]))
            smooth_rot = slerp(np.linspace(0, 1, cache_size + 1)).as_euler('XYZ')[1]
            ## using NLerp
            # mean_quat = Rotation.from_euler('XYZ', pred_euler_list).as_quat().mean(0)
            # quats_norm = np.linalg.norm(mean_quat) + 1e-6
            # smooth_rot = Rotation.from_quat(mean_quat / quats_norm).as_euler('XYZ')[1]
            ## using chordal L2 mean
            # smooth_rot = Rotation.from_euler('XYZ', pred_euler_list).mean().as_euler('XYZ')[1]

        if self.config.use_delta:
            if self.config.smooth:
                delta_pos, delta_rot = smooth_pos, smooth_rot
            else:
                delta_pos, delta_rot = action[0:3], action[3:6]
            delta_quat = Rotation.from_euler('XYZ', delta_rot).as_quat()[[3, 0, 1, 2]]
            delta_pose = sapien.Pose(delta_pos, delta_quat)

            if self.config.frame == "base":
                target_pose = delta_pose * prev_ee_pose_at_base
            elif self.config.frame == "ee":
                target_pose = prev_ee_pose_at_base * delta_pose
            elif self.config.frame == "ee_align":
                # origin at ee but base rotation
                target_pose = delta_pose * prev_ee_pose_at_base
                target_pose.set_p(prev_ee_pose_at_base.p + delta_pos)
            else:
                raise NotImplementedError(self.config.frame)
        else:
            assert self.config.frame == "base", self.config.frame
            target_pos, target_rot = action[0:3], action[3:6]
            target_quat = Rotation.from_euler('XYZ', target_rot).as_quat()[[3, 0, 1, 2]]
            target_pose = sapien.Pose(target_pos, target_quat)

        return target_pose


@dataclass
class PDEEPoseEulerControllerMultiActConfig(ControllerConfig):
    pos_lower: Union[float, Sequence[float]]
    pos_upper: Union[float, Sequence[float]]
    rot_bound: float
    stiffness: Union[float, Sequence[float]]
    damping: Union[float, Sequence[float]]
    force_limit: Union[float, Sequence[float]] = 1e10
    friction: Union[float, Sequence[float]] = 0.0
    cache_size: int = 3
    ee_link: str = None
    frame: str = "ee"  # [base, ee, ee_align]
    smooth: bool = False
    use_delta: bool = True
    use_target: bool = False
    interpolate: bool = False
    normalize_action: bool = True
    controller_cls = PDEEPoseEulerControllerMultiAct
