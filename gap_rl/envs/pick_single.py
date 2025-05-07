from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Type, Union

import numpy as np
import torch
import open3d as o3d
import sapien.core as sapien
from gap_rl import ASSET_DIR, format_path
from gap_rl.agents.base_agent import BaseAgent
from gap_rl.agents.robots.ur5e_robotiq85_old import UR5e_Robotiq85_old
from gap_rl.agents.robots.indy7_robotiq85_old import Indy7_Robotiq85_old
from gap_rl.sensors.camera import CameraConfig
from gap_rl.utils.common import (
    convert_np_bool_to_float,
    flatten_state_dict,
    random_choice,
)
from gap_rl.utils.geometry import (
    angle_distance_ms,
    mat_to_posrotvec,
    pc_bbdx_filter,
    pointcloud_filter,
    pose_to_posrotvec,
    sample_grasp_multipoints_ee,
    sample_grasp_points_ee,
    sample_query_grasp_points_ee,
    sample_grasp_keypoints_ee,
    transform_points,
    xyz2uvz,
)
from gap_rl.utils.io_utils import load_json
from gap_rl.utils.registration import register_env
from gap_rl.utils.sapien_utils import (
    compute_total_impulse,
    get_entity_by_name,
    get_pairwise_contacts,
    look_at,
    set_actor_visibility,
    set_articulation_render_material,
    vectorize_pose,
)
from gap_rl.utils.traj_utils import gen_traj
from gap_rl.utils.trimesh_utils import (
    get_actor_mesh,
    get_actor_meshes,
    get_actor_visual_mesh,
    get_actor_visual_meshes,
    get_articulation_meshes,
    merge_meshes,
)
from sapien.core import Pose
from sapien.sensor.stereodepth import StereoDepthSensor
from scipy.spatial.transform import Rotation
from transforms3d.euler import euler2quat, quat2euler
from transforms3d.quaternions import axangle2quat, qmult

from gap_rl.envs.base_env import BaseEnv


@register_env("PickSingle-v0", max_episode_steps=100)
class PickSingleEnv(BaseEnv):
    DEFAULT_ASSET_ROOT: str
    DEFAULT_MODEL_JSON: str
    DEFAULT_GRASP_JSON: str

    SUPPORTED_OBS_MODES = (
        "none",
        "state_egopoints",
        "state_egopoints_rt",
        "state_objpoints_rt",
        "state_grasp9d",
        "state_grasp9d_rt",
    )
    SUPPORTED_REWARD_MODES = ("dense", "sparse")
    SUPPORTED_ROBOTS = {
        "ur5e_robotiq85_old": UR5e_Robotiq85_old,
        "indy7_robotiq85_old": Indy7_Robotiq85_old,
    }
    agent: Union[UR5e_Robotiq85_old]

    obj: sapien.Actor  # target object

    def __init__(
        self,
        robot="panda",
        robot_init_qpos_noise=0.05,
        asset_root: str = None,
        model_json: str = None,
        num_grasps: int = 10,
        num_grasp_points: int = 20,
        grasp_points_mode: str = "gauss",  # keypoints, gauss, gauss_fix, uniform
        model_ids: List[str] = (),
        obj_init_rot_z: bool = True,
        obj_init_rot: float = 0.0,
        goal_thresh: float = 0.25,
        goal_pos: List[float] = [0.5, 0.0, 0.3],
        robot_x_offset: float = 0.56,
        gen_traj_mode: str = None,
        **kwargs,
    ):
        if asset_root is None:
            asset_root = self.DEFAULT_ASSET_ROOT
        self.asset_root = Path(format_path(asset_root))

        if model_json is None:
            model_json = self.DEFAULT_MODEL_JSON
        model_json = self.asset_root / format_path(model_json)

        if not model_json.exists():
            raise FileNotFoundError(
                f"{model_json} is not found."
                "Please download the corresponding assets:"
                "`python -m gap_rl.utils.download_asset ${ENV_ID}`."
            )
        self.model_db: Dict[str, Dict] = load_json(model_json)

        self.num_grasps = num_grasps

        if isinstance(model_ids, str):
            model_ids = [model_ids]
        if len(model_ids) == 0:
            model_ids = sorted(self.model_db.keys())
        assert len(model_ids) > 0, model_json
        self.model_ids = model_ids

        self.model_id = model_ids[0]
        self.model_scale = None
        self.model_bbox_size = None

        self.obj_init_rot_z = obj_init_rot_z
        self.obj_init_rot = obj_init_rot
        self.goal_thresh = goal_thresh
        self.goal_pos = goal_pos
        self.robot_x_offset = robot_x_offset
        self.gen_traj_mode = gen_traj_mode
        self.vary_speed = kwargs.pop("vary_speed", False)

        self.contact_obj_pose = Pose()
        self.cam_paras = OrderedDict()

        self.robot_uid = robot
        self.gripper_w = 0.0425 if "85" in robot else 0.068
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.contact_flag = False
        self.grasps_mat = None

        assert grasp_points_mode in ["gauss", "keypoints", "gauss_fix", "uniform"], "not support mode! "
        self.grasp_points_mode = grasp_points_mode
        self.num_grasp_points = num_grasp_points
        self.gripper_pts = sample_grasp_points_ee(
            [self.gripper_w, self.gripper_w], z_offset=0.02
        )  # gripper keypoints, (6, 3)
        self.gripper_pts_rect = sample_grasp_keypoints_ee(
            gripper_w=self.gripper_w,
            num_points_perlink=int(self.num_grasp_points / 4)
        )
        fix_gauss_rng = np.random.RandomState(1029)
        self.gripper_pts_gauss = fix_gauss_rng.normal(0.0, self.gripper_w/3, size=(num_grasp_points, 3))
        self.grasps_mat_ee = np.zeros((num_grasps, 4, 4))
        self.grasps_scores = np.zeros(num_grasps)
        self.pred_grasp_actor_critic = None
        # angle_filter, nearest, random, near4, near4_filter
        self.grasp_select_mode = kwargs.pop("grasp_select_mode", "angle_filter")
        self.camera_modes = ["hand_realsense"]
        self.points_add_noise = False

        self._cache_info = {}
        self.device = torch.device(kwargs.pop("device", "cuda:0"))

        self._check_assets()
        super().__init__(**kwargs)

    def _check_assets(self):
        """Check whether the assets exist."""
        pass

    def _build_sphere_site(self, radius, color=(0, 1, 0), name="goal_site"):
        """Build a sphere site (visual only)."""
        builder = self._scene.create_actor_builder()
        builder.add_sphere_visual(radius=radius, color=color)
        sphere = builder.build_static(name)
        sphere.hide_visual()
        return sphere

    def _build_box(
        self,
        pose: sapien.Pose,
        phy_mat: None,
        half_size=(0.05, 0.05, 0.01),
        density=1e3,
        color=(0, 0, 0),
        name="",
        hide_visual=True,
    ):
        builder = self._scene.create_actor_builder()
        builder.add_box_visual(pose=Pose(), half_size=half_size, color=color)
        builder.add_box_collision(
            half_size=half_size, material=phy_mat, density=density
        )
        box = builder.build(name=name)
        if hide_visual:
            box.hide_visual()
        box.set_pose(pose)
        return box

    def _load_actors(self):
        self._add_ground(render=self.bg_name is None)
        self._load_model()
        self.obj.set_damping(0.1, 0.1)
        half_height = 0.005
        self.box_halfsize = (self.obj_aabb_halfsize[:2] + 0.03).tolist() + [half_height]
        phy_mat = self._scene.create_physical_material(
            static_friction=1, dynamic_friction=1, restitution=0.0
        )
        self.drive_base = self._build_box(
            pose=Pose([self.robot_x_offset, 0, self.box_halfsize[2]]),
            phy_mat=phy_mat,
            half_size=self.box_halfsize,
            density=1e6,
            color=(0, 0, 0),
            name="drive_base",
            hide_visual=False,
        )

        self.table_halfsize = [0.4, 1.0, 0.04]
        self.table_halfsize[2] += self._episode_rng.uniform(-0.02, 0.02)
        self.table = self._build_box(
            pose=Pose([self.robot_x_offset, 0, self.table_halfsize[2]]),
            phy_mat=phy_mat,
            half_size=self.table_halfsize,
            density=1e3,
            color=(0.2, 0.2, 0.2),
            name="table",
            hide_visual=False,
        )
        ## drive(conveyor, obj)
        drive_base_pos = self.drive_base.pose.p
        p_obj = drive_base_pos + np.array(
            [0, 0, self._get_init_z() + drive_base_pos[2]]
        )
        pose_d = self.drive_base.pose.inv() * Pose(p_obj)
        self.conveyor_drive = self._scene.create_drive(
            self.drive_base, pose_d, self.obj, Pose()
        )
        self.conveyor_drive.lock_motion(False, False, False, False, False, False)

    def _load_model(self):
        """Load the target object."""
        raise NotImplementedError

    def _configure_agent(self):
        agent_cls: Type[BaseAgent] = self.SUPPORTED_ROBOTS[self.robot_uid]
        self._agent_cfg = agent_cls.get_default_config()

    def _load_agent(self):
        agent_cls: Type[Panda] = self.SUPPORTED_ROBOTS[self.robot_uid]
        self.agent = agent_cls(
            self._scene, self._control_freq, self._control_mode, config=self._agent_cfg
        )
        self.tcp: sapien.Link = get_entity_by_name(
            self.agent.robot.get_links(), self.agent.config.ee_link_name
        )
        set_articulation_render_material(self.agent.robot, specular=0.9, roughness=0.3)
        if self.robot_uid == "panda":
            pass
        elif self.robot_uid == "xmate3_robotiq":
            qpos = np.array(
                [0, -np.pi / 6, 0, np.pi / 2, 0, 7 * np.pi / 12, np.pi / 2,
                 0, 0, 0, 0, 0, 0]
            )
            self.agent.reset(qpos)
            self.agent.robot.set_pose(Pose())
            self.agent._add_constraints()
        elif self.robot_uid == "ur5e_robotiq140":
            qpos = np.array(
                [-1.1672, -0.9103, -1.8854, -1.5976, 1.4559, -2.7273, 0, 0, 0, 0, 0, 0]
            )
            self.agent.reset(qpos)
            self.agent.robot.set_pose(Pose())
            self.agent._add_constraints()
        elif self.robot_uid == "ur5e_robotiq85":
            qpos = np.array(
                [-1.1672, -0.9103, -1.8854, -1.5976, 1.4559, -2.7273, 0, 0, 0, 0, 0, 0]
            )
            self.agent.reset(qpos)
            self.agent.robot.set_pose(Pose())
            self.agent._add_constraints()
        elif self.robot_uid == "indy7_robotiq85_old":
            pass
        elif self.robot_uid in ["ur5e_robotiq140_old", "ur5e_robotiq85_old"]:
            pass
        else:
            raise NotImplementedError(self.robot_uid)
        


        if self.robot_uid == "panda" or self.robot_uid == "xmate3_robotiq":
            self.num_joints = 7
        elif self.robot_uid in ["ur5e_robotiq140", "ur5e_robotiq140_old", "ur5e_robotiq85", "ur5e_robotiq85_old"]:
            self.num_joints = 6
        elif self.robot_uid == "indy7_robotiq85_old":
            self.num_joints = 6
        else:
            raise NotImplementedError

    # BK fix: options 추가
    # gym 0.23.0 stable_baselinses3 최신 버전 python 3.10쓰기 위함
    def reset(self, seed=None, reconfigure=False, model_id=None, model_scale=None, options=None):
        self.set_episode_rng(seed)
        _reconfigure = self._set_model(model_id, model_scale)
        reconfigure = _reconfigure or reconfigure
        self._cache_info.clear()
        self.dynamic_paras = None
        self.grasp_ids = None
        self.grasps_mat_ee = np.zeros((self.num_grasps, 4, 4))
        self.grasps_scores = np.zeros(self.num_grasps)
        self.pred_grasp_actor_critic = None
        self.pred_target_actor_critic = np.zeros(4)
        obs = super().reset(seed=self._episode_seed, reconfigure=reconfigure)

        info = {}  # BK fig
        return obs, info

    def _set_model(self, model_id, model_scale):
        """Set the model id and scale. If not provided, choose one randomly."""
        reconfigure = False

        if model_id is None:
            model_id = random_choice(self.model_ids, self._episode_rng)
        if model_id != self.model_id:
            self.model_id = model_id
            reconfigure = True

        if model_scale is None:
            model_scales = self.model_db[self.model_id].get("scales")
            if model_scales is None:
                model_scale = 1.0
            else:
                model_scale = random_choice(model_scales, self._episode_rng)
        if model_scale != self.model_scale:
            self.model_scale = model_scale
            reconfigure = True

        model_info = self.model_db[self.model_id]
        if "bbox" in model_info:
            bbox = model_info["bbox"]
            bbox_size = np.array(bbox["max"]) - np.array(bbox["min"])
            self.model_bbox_size = bbox_size * self.model_scale
        else:
            self.model_bbox_size = None

        return reconfigure

    def _get_init_z(self):
        return 0.5

    def _get_obj_init_xyz(self, drive_base_p):
        obj_p = drive_base_p + np.array(
            [0, 0, self._get_init_z() + self.box_halfsize[2]]
        )
        return obj_p

    def _settle(self, t):
        sim_steps = int(self.sim_freq * t)
        for _ in range(sim_steps):
            self._scene.step()

    def _initialize_actors(self):
        if not self.gen_traj_mode:
            # The object will fall from a certain height
            xy = self._episode_rng.uniform(-0.1, 0.1, [2])
            xy += np.array([self.robot_x_offset, 0])

            # set conveyor pose & obj pose
            drive_base_p = np.hstack([xy, self.box_halfsize[2]])
            self.drive_poses = None
        else:
            if self.dynamic_paras is None:
                if self.gen_traj_mode == "line":
                    dist = float(self._episode_rng.uniform(0.55, 0.65))
                    angle = float(self._episode_rng.uniform(-np.pi / 6, np.pi / 6))
                    speed = float(self._episode_rng.uniform(0.01, 0.06))
                elif self.gen_traj_mode == "circle":
                    dist = float(self._episode_rng.uniform(0.55, 0.65))
                    angle = float(self._episode_rng.uniform(-np.pi / 6, np.pi / 6))
                    speed = float(self._episode_rng.uniform(0.01, 0.1))
                elif self.gen_traj_mode == "circular":
                    dist = float(self._episode_rng.uniform(0.55, 0.65))
                    speed = float(self._episode_rng.uniform(0.01, 0.3))
                    angle = float(self._episode_rng.uniform(0.1, 0.2))
                elif self.gen_traj_mode == "bezier2d":
                    dist = 0.6
                    speed = float(self._episode_rng.uniform(0.01, 0.06))
                    angle = float(self._episode_rng.uniform(0.18, 0.22))
                elif self.gen_traj_mode == "random2d":
                    dist = 0.6
                    speed = 0
                    angle = float(self._episode_rng.uniform(0.18, 0.22))
                else:
                    raise NotImplementedError
                length = float(speed * 20) + 0.01  # max_episode_steps / control_freq
                self.dynamic_paras = dict(
                    speed=speed,
                    dist=dist,
                    angle=angle,
                    length=length,
                    rotz_inv=random_choice([True, False], self._episode_rng),
                    vary_speed=self.vary_speed,
                )
            traj_poses = gen_traj(
                rng=self._episode_rng,
                dynamic_paras=deepcopy(self.dynamic_paras),
                traj_mode=self.gen_traj_mode,
                sim_freq=self.sim_freq,
                control_freq=self.control_freq,
                max_steps=100,
            )  # (N, 3)
            traj_poses = np.concatenate(
                (np.repeat(traj_poses[0][None], 30, axis=0), traj_poses)
            )
            drive_poses = traj_poses + np.array(
                [0, 0, self.table_halfsize[2] * 2 + self.box_halfsize[2]]
            )
            self.drive_poses = drive_poses

        ori = 0
        q = [1, 0, 0, 0]
        # Rotate along z-axis
        if self.obj_init_rot_z:
            ori = self._episode_rng.uniform(0, 2 * np.pi)
            q = euler2quat(0, 0, ori)
        # Rotate along a random axis by a small angle
        if self.obj_init_rot > 0:
            axis = self._episode_rng.uniform(-1, 1, 3)
            axis = axis / max(np.linalg.norm(axis), 1e-6)
            ori = self._episode_rng.uniform(0, self.obj_init_rot)
            q = qmult(q, axangle2quat(axis, ori, True))
        self.init_q = q

        # rotate drive base
        if self.vary_speed:
            rot_v = self._episode_rng.uniform(-np.pi / 18, np.pi / 18) / self.sim_freq
            # max 100 steps
            euler_segs = (
                np.arange(100 / self.control_freq * self.sim_freq) * rot_v + ori
            )
            self.qs = np.roll(Rotation.from_euler("z", euler_segs).as_quat(), 1, axis=1)

        ## new drive
        drive_base_p = self.drive_poses[0]
        self.drive_base.set_pose(Pose(p=drive_base_p, q=self.init_q))
        obj_p = self._get_obj_init_xyz(drive_base_p)
        self.obj.set_pose(Pose(obj_p, self.init_q))

        # Some objects need longer time to settle
        lin_vel = np.linalg.norm(self.obj.velocity)
        ang_vel = np.linalg.norm(self.obj.angular_velocity)
        if lin_vel > 1e-3 or ang_vel > 1e-2:
            self._settle(0.5)

        # get object init position
        self.obj_init_pos = self.obj.pose.p

    def _initialize_agent(self):
        if self.robot_uid == "ur5e_robotiq85_old":
            # qpos = np.array(
            #     [-1.27 , -0.9, -1.6, -1.7, 1.4, -2.7, 0, 0]  # top-down view
            # )
            qpos = np.array(
                [-1.1672, -0.9103, -1.8854, -1.5976, 1.4559, -2.7273, 0, 0]  # top-down view
            )
            qpos[:-2] += self._episode_rng.normal(
                0, self.robot_init_qpos_noise, len(qpos) - 2
            )
            self.agent.reset(qpos)
            self.agent.robot.set_pose(Pose())
        elif self.robot_uid == "indy7_robotiq85_old":
            # qpos = np.array(
            #     [-1.27, -0.9, -1.6, -1.7, 1.4, -2.7, 0, 0]  # top-down view
            # )
            qpos = np.array(
                [0.5, 0.0, -1.0, 0.0, -1.7, 0.0, 0, 0]  # top-down view
            )
            qpos[:-2] += self._episode_rng.normal(
                0, self.robot_init_qpos_noise, len(qpos) - 2
            )
            self.agent.reset(qpos)
            self.agent.robot.set_pose(Pose())

        else:
            raise NotImplementedError(self.robot_uid)
        self._get_cam_info(cam_name="hand_realsense")
        # get cam2ee transform
        handcam_joint = get_entity_by_name(self.agent.robot.get_joints(), "realsense_hand_joint")
        self.trans_cam2ee = handcam_joint.get_pose_in_parent().to_transformation_matrix()

    @property
    def obj_pose(self):
        """Get the center of mass (COM) pose."""
        return self.obj.pose.transform(self.obj.cmass_local_pose)

    def _get_cam_extrins(self, cam_name):
        self.update_render()
        # world2cam, <=> camera_hand_color_frame.pose.inv().to_transformation_matrix()
        if isinstance(self._cameras["hand_realsense"].camera, StereoDepthSensor):
            self.cam_paras[cam_name + "_extrinsic"] = self._cameras[
                "hand_realsense"
            ].camera._cam_rgb.get_extrinsic_matrix()
        else:
            self.cam_paras[cam_name + "_extrinsic"] = self._cameras[
                "hand_realsense"
            ].camera.get_extrinsic_matrix()

    def _get_cam_info(self, cam_name):
        if isinstance(self._cameras["hand_realsense"].camera, StereoDepthSensor):
            self.cam_paras[cam_name + "_intrinsic"] = self._cameras[
                "hand_realsense"
            ].camera._cam_rgb.get_intrinsic_matrix()
        else:
            self.cam_paras[cam_name + "_intrinsic"] = self._cameras[
                "hand_realsense"
            ].camera.get_intrinsic_matrix()
        self.cam_paras[cam_name + "_height"] = self._camera_cfgs[cam_name].height
        self.cam_paras[cam_name + "_width"] = self._camera_cfgs[cam_name].width
        self._get_cam_extrins(cam_name)

    def _initialize_task(self, max_trials=100):
        # pass
        self.contact_flag = False
        self.conveyor_drive.set_x_properties(stiffness=1e6, damping=1e3)
        self.conveyor_drive.set_y_properties(stiffness=1e6, damping=1e3)
        self.conveyor_drive.set_z_properties(stiffness=1e8, damping=1e4)
        self.conveyor_drive.set_slerp_properties(stiffness=1e6, damping=1e3)

    def _compute_near_grasps(self):
        # angle_filter, nearest, random, near4, near4_filter
        if self.grasp_select_mode in ["near4", "near4_filter", "nearest"]:
            cam_pose = self._cameras['hand_realsense'].camera.get_model_matrix()  # cam2world
            obj_tcp_vec = cam_pose[:3, 3] - self.obj_pose.p
            dist = np.linalg.norm(obj_tcp_vec)
            obj_rot_mat = self.obj_pose.to_transformation_matrix()[:3, :3]
            x, y, z = obj_rot_mat[:, 0], obj_rot_mat[:, 1], obj_rot_mat[:, 2]
            # angle between tcp & xy plane
            vec_z_dot = np.dot(obj_tcp_vec, z)
            theta = np.pi / 2 - np.arccos(vec_z_dot / dist)

            obj_tcp_vec_proj_z = vec_z_dot * z
            obj_tcp_vec_proj_xy = obj_tcp_vec - obj_tcp_vec_proj_z
            dist_xy = np.linalg.norm(obj_tcp_vec_proj_xy) + 0.001
            vecxy_x_dot = np.dot(obj_tcp_vec_proj_xy, x)
            if np.dot(obj_tcp_vec_proj_xy, y) > 0:
                phi = np.pi - np.arccos(vecxy_x_dot / dist_xy)
            else:
                phi = np.arccos(vecxy_x_dot / dist_xy) - np.pi

            phi_min, phi_max = -180, 180 * 5 / 6
            phi_d = (phi_max - phi_min) // 11
            phi_t = (phi / np.pi * 180 - phi_min) / phi_d  # [0, 12)
            # phi_t = (phi / np.pi + 1) * 6  # [0, 12)
            theta_min, theta_max = 10, 80
            theta_d = (theta_max - theta_min) // 5
            theta_t = (theta / np.pi * 180 - theta_min) / theta_d  # [0, 5)

            if self.grasp_select_mode == "nearest":
                phi_id = round(phi_t) % 12
                theta_id = max(0, min(round(theta_t), 5))
                grasp_view_ids = [phi_id*6 + theta_id]
            else:
                phi_ids = [int(phi_t), (int(phi_t) + 1) % 12]
                if theta_t <= 0:
                    theta_ids = [0, 1]
                elif theta_t < 5:
                    theta_ids = [int(theta_t), int(theta_t) + 1]
                else:
                    theta_ids = [4, 5]
                grasp_view_ids = [phi_ids[0]*6 + theta_ids[0], phi_ids[0]*6 + theta_ids[1],
                                  phi_ids[1]*6 + theta_ids[0], phi_ids[1]*6 + theta_ids[1]]

            lg_grasps_poses = []
            lg_grasps_scores = []


            for grasp_view_id in grasp_view_ids:
                if self.lg_grasps_dict[grasp_view_id] is not None:
                    transformations = self.lg_grasps_dict[grasp_view_id]['transformations']
                    lg_grasps_poses.extend(transformations)
                    scores = self.lg_grasps_dict[grasp_view_id]['scores']
                    lg_grasps_scores.extend(scores)


            if len(lg_grasps_poses) > 0:
                lg_grasps_poses = np.array(lg_grasps_poses)
                grasp_mats = np.repeat(np.eye(4)[None], lg_grasps_poses.shape[0], 0)
                grasp_mats[:, :3, 3] = lg_grasps_poses[:, :3]
                grasp_mats[:, :3, :3] = Rotation.from_quat(lg_grasps_poses[:, 3:]).as_matrix()
                self.lg_grasps_mat = grasp_mats  # (M, 4, 4)
                self.lg_grasps_score = np.array(lg_grasps_scores)
            else:
                self.lg_grasps_mat = np.zeros((1, 4, 4))
                self.lg_grasps_score = np.zeros(1)

        ### get grasps of the obj and trans to EE frame
        grasp_ids = np.arange(self.lg_grasps_mat.shape[0])

        # Grasp 후보 (self.lg_grasps_mat)를 object 좌표계 => ee 좌표계로 변환
        ## transfer to ee frame
        trans_obj2ee = self._get_obj2ee()
        grasps_mat_ee = np.einsum(
            "ij, kjl -> kil", trans_obj2ee, self.lg_grasps_mat
        )  # (N, 4, 4)

        ## ego-view filter
        grasp_exist_mask = self._get_grasp_exist_mask()
        grasp_ids = grasp_ids[grasp_exist_mask]

        if "filter" in self.grasp_select_mode:
            ## filter the grasp (<direction, grasp direction> > 60 deg)
            direct_mask = grasps_mat_ee[grasp_ids, 2, 2] > np.cos(np.pi / 3)  # 60 or 90
            grasp_ids = grasp_ids[direct_mask]

        ## 0~K grasps randomly
        # sparse_grasp_nums = self._episode_rng.randint(0, len(grasp_ids)) if len(grasp_ids) > 0 else 0
        # self.grasp_ids = self._episode_rng.choice(grasp_ids, min(sparse_grasp_nums, self.num_grasps), replace=False)
        ## keep all grasps
        if len(grasp_ids) > self.num_grasps:
            self.grasp_ids = self._episode_rng.choice(
                grasp_ids, self.num_grasps, replace=False
            )
        else:
            self.grasp_ids = grasp_ids

        grasps_ee = np.zeros((self.num_grasps, 4, 4))
        valid_grasp_num = len(self.grasp_ids)
        grasps_ee[:valid_grasp_num] = grasps_mat_ee[self.grasp_ids]

        ## add noise to grasp center
        # grasps_ee[:valid_grasp_num, :3, 3] += self._episode_rng.normal(0, 0.005, (valid_grasp_num, 3))

        grasps_scores = np.zeros(self.num_grasps)
        grasps_scores[:valid_grasp_num] = self.lg_grasps_score[self.grasp_ids]

        return grasps_ee, grasps_scores

    def _get_state_egopoints(self, action) -> OrderedDict:
        agent_qpos = self.agent.robot.get_qpos()
        cur_ee_pos, cur_ee_quat = self.tcp.pose.p, self.tcp.pose.q
        cur_ee_euler = Rotation.from_quat(np.roll(cur_ee_quat, -1)).as_euler("XYZ")

        obs = OrderedDict(
            gripper_pos=agent_qpos[self.agent.gripper_joint_ids].astype(np.float32),
            tcp_pose=np.hstack([cur_ee_pos, cur_ee_euler]).astype(np.float32),
            action=action.astype(np.float32),
        )

        # github에 있는 Local Grasp을 매번 inference 하는 대신에, 미리 저장해놓고 근사하는 코드.
        ### compute current nearest LoG grasps
        grasps_ee, grasps_scores = self._compute_near_grasps()

        ## add grasp mask in hand_camera
        grasp_exist = np.ones(5) if len(self.grasp_ids) > 0 else np.zeros(5)

        # IV-D. 1) Grasp as points: {x_i}_i=1^L 뽑는 과정 (20개 뽑는다)
        if self.grasp_points_mode == "gauss":
            ## gaussian points
            rng = np.random.RandomState(np.random.RandomState().randint(2 ** 32))
            gripper_pts_ee = rng.normal(0.0, self.gripper_w / 3, size=(self.num_grasp_points, 3))
        elif self.grasp_points_mode == "gauss_fix":
            ## fixed gaussian points
            gripper_pts_ee = self.gripper_pts_gauss
        elif self.grasp_points_mode == "uniform":
            ## uniformly sampled points
            rng = np.random.RandomState(np.random.RandomState().randint(2 ** 32))
            gripper_pts_ee = rng.uniform(-self.gripper_w, self.gripper_w, size=(self.num_grasp_points, 3))
        elif self.grasp_points_mode == "keypoints":
            ## rigid set of keypoints
            gripper_pts_ee = self.gripper_pts_rect
        else:
            raise NotImplementedError

        # IV-D. 1) Grasp as points: T_Xh를 적용하는 과정
        R, T = grasps_ee[:, :3, :3].transpose((0, 2, 1)), grasps_ee[:, :3, 3]
        gripper_pts_diff = np.einsum("ij, kjl -> kil", gripper_pts_ee, R) + np.repeat(
            T[:, None, :], gripper_pts_ee.shape[0], axis=1
        )  # (num, k, 3)

        if self.grasps_mat is not None:
            close_grasp_pose_ee, _ = self._get_closest_grasp_pose()
        else:
            # close_grasp_pose_ee = np.zeros(7)
            close_grasp_pose_ee = np.zeros(9)

        ## self-supervised reward target
        eval_target = np.array(self.evaluate_success())

        obs.update(
            origin_gripper_pts=gripper_pts_ee.astype(np.float32),
            grasp_exist=grasp_exist.astype(np.float32),
            gripper_pts_diff=gripper_pts_diff.astype(np.float32),
            grasps_scores=grasps_scores.astype(np.float32),
            close_grasp_pose_ee=close_grasp_pose_ee.astype(np.float32),
            eval_target=eval_target.astype(np.float32),
        )
        return obs

    def _get_state_grasp9d(self, action) -> OrderedDict:
        agent_qpos = self.agent.robot.get_qpos()
        cur_ee_pos, cur_ee_quat = self.tcp.pose.p, self.tcp.pose.q
        cur_ee_euler = Rotation.from_quat(np.roll(cur_ee_quat, -1)).as_euler("XYZ")
        obs = OrderedDict(
            gripper_pos=agent_qpos[self.agent.gripper_joint_ids].astype(np.float32),
            tcp_pose=np.hstack([cur_ee_pos, cur_ee_euler]).astype(np.float32),
            action=action.astype(np.float32),
        )

        ### compute current nearest LoG grasps
        grasps_ee, grasps_scores = self._compute_near_grasps()

        ## add grasp mask in hand_camera
        grasp_exist = np.ones(5) if len(self.grasp_ids) > 0 else np.zeros(5)

        grasps_posrot_ee = np.concatenate(
            (grasps_ee[:, :3, 3], grasps_ee[:, :3, 0], grasps_ee[:, :3, 1]), axis=1
        )  # (num, 9)

        if self.grasps_mat is not None:
            close_grasp_pose_ee, _ = self._get_closest_grasp_pose()
        else:
            # close_grasp_pose_ee = np.zeros(7)
            close_grasp_pose_ee = np.zeros(9)

        obs.update(
            grasp_exist=grasp_exist.astype(np.float32),
            grasps_posrot_ee=grasps_posrot_ee.astype(np.float32),
            grasps_scores=grasps_scores.astype(np.float32),
            close_grasp_pose_ee=close_grasp_pose_ee.astype(np.float32),
        )
        return obs

    def _get_obj_exist_mask(self):
        ## add grasp mask in hand_camera
        self._get_cam_extrins("hand_realsense")
        handcam_intrin = self.cam_paras["hand_realsense_intrinsic"]
        trans_world2cam = self.cam_paras["hand_realsense_extrinsic"]
        trans_obj2world = self.obj_pose.to_transformation_matrix()
        trans_obj2cam = trans_world2cam @ trans_obj2world
        obj_pc_cam = transform_points(trans_obj2cam, self.obj_pc)
        obj_pc_uvz = xyz2uvz(obj_pc_cam, handcam_intrin)
        obj_exist_mask = (
            (obj_pc_uvz[:, 0] > 0)
            & (obj_pc_uvz[:, 0] < self.cam_paras["hand_realsense_width"])
            & (obj_pc_uvz[:, 1] > 0)
            & (obj_pc_uvz[:, 1] < self.cam_paras["hand_realsense_height"])
        )
        return obj_exist_mask

    # 특정 시야에서 특정 grasp가 보이는지 안보이는지
    def _get_grasp_exist_mask(self):
        if self.obs_mode == "state_egortgrasppoints":
            return 1 if self.grasps_mat_ee.shape[0] > 1 else 0
        self._get_cam_extrins("hand_realsense")
        handcam_intrin = self.cam_paras["hand_realsense_intrinsic"]
        trans_world2cam = self.cam_paras["hand_realsense_extrinsic"]
        if self.obs_mode in ["state_egopoints", "state_grasp9d", "state_grasp_obj_points"]:
            trans_obj2world = self.obj_pose.to_transformation_matrix()
            trans_obj2cam = trans_world2cam @ trans_obj2world
            grasps_centers_cam = transform_points(
                trans_obj2cam, self.lg_grasps_mat[:, :3, 3]
            )
        elif self.obs_mode in ["state_egopoints_rt", "state_grasp9d_rt", "state_grasp_obj_points_rt"]:
            trans_ee2world = self.tcp.pose.to_transformation_matrix()
            trans_ee2cam = trans_world2cam @ trans_ee2world
            grasps_centers_cam = transform_points(
                trans_ee2cam, self.grasps_mat_ee[:, :3, 3]
            )
        else:
            raise NotImplementedError
        grasps_centers_uvz = xyz2uvz(grasps_centers_cam, handcam_intrin)
        grasp_exist_mask = (
            (grasps_centers_uvz[:, 0] > 0)
            & (grasps_centers_uvz[:, 0] < self.cam_paras["hand_realsense_width"])
            & (grasps_centers_uvz[:, 1] > 0)
            & (grasps_centers_uvz[:, 1] < self.cam_paras["hand_realsense_height"])
        )
        return grasp_exist_mask

    def _get_obj2ee(self):
        trans_obj2world = self.obj_pose.to_transformation_matrix()
        trans_world2ee = self.tcp.pose.inv().to_transformation_matrix()
        trans_obj2ee = trans_world2ee @ trans_obj2world
        return trans_obj2ee

    def _get_closest_grasp_pose(self):
        trans_obj2ee = self._get_obj2ee()

        ## using trans + quat dist
        grasps_mat_ee = np.einsum(
            "ij, kjl -> kil", trans_obj2ee, self.grasps_mat
        )  # (N, 4, 4)
        translation_dist = np.linalg.norm(grasps_mat_ee[:, :3, 3], axis=1)  # (N,)
        quat = Rotation.from_matrix(
            grasps_mat_ee[:, :3, :3]
        ).as_quat()  # (N, 4) xyzw
        rot_dist0 = 1 - np.clip(np.abs(quat[:, 2]), a_min=0, a_max=1)
        rot_dist1 = 1 - np.clip(np.abs(quat[:, 3]), a_min=0, a_max=1)
        rotation_dist = np.minimum(rot_dist0, rot_dist1)  # (N,)
        translation_dist = np.clip(translation_dist, a_min=0.01, a_max=0.5)
        rotation_dist = np.clip(rotation_dist, a_min=0.01, a_max=0.5)
        grasp_dist = translation_dist + rotation_dist
        min_id = np.argmin(grasp_dist)

        closest_grasp_mat_ee = grasps_mat_ee[min_id]
        if self.obs_mode == "state_objpoints_rt":
            closest_grasp_pose_ee = np.zeros(7)
            closest_grasp_pose_ee[:4] = np.roll(Rotation.from_matrix(closest_grasp_mat_ee[:3, :3]).as_quat(), 1)  # wxyz
            closest_grasp_pose_ee[4:] = closest_grasp_mat_ee[:3, 3]
        else:
            closest_grasp_pose_ee = np.zeros(9)
            closest_grasp_pose_ee[:6] = np.hstack((closest_grasp_mat_ee[:3, 0], closest_grasp_mat_ee[:3, 1]))  # rx, ry
            closest_grasp_pose_ee[6:] = closest_grasp_mat_ee[:3, 3]

        return closest_grasp_pose_ee, min_id

    def get_obs(self, action: np.ndarray):
        if self._obs_mode == "none":
            # Some cases do not need observations, e.g., MPC
            return OrderedDict()
        elif self._obs_mode == "state_egopoints":
            return self._get_state_egopoints(action)
        elif self._obs_mode == "state_egopoints_rt":
            return self.get_state_egopoints_rt(action)
        elif self._obs_mode == "state_grasp9d":
            return self._get_state_grasp9d(action)
        elif self._obs_mode == "state_grasp9d_rt":
            return self.get_state_grasp9d_rt(action)
        elif self._obs_mode == "state_objpoints_rt":
            return self.get_state_objpoints_rt(action)
        else:
            raise NotImplementedError(self._obs_mode)

    def set_rt_paras(self, **kwargs):
        self.camera_modes = kwargs.pop("camera_modes", self.camera_modes)
        self.points_add_noise = kwargs.pop("add_noise", self.points_add_noise)
        self.grasps_mat_ee = kwargs.pop("grasps_mat_ee", self.grasps_mat_ee)
        self.grasps_scores = kwargs.pop("grasps_scores", self.grasps_scores)
        self.pred_grasp_actor_critic = kwargs.pop("pred_grasp_actor_critic", self.pred_grasp_actor_critic)
        self.pred_target_actor_critic = kwargs.pop("pred_target_actor_critic", self.pred_target_actor_critic)

    def get_objpoints_rt(self):
        trans_world2ee = self.tcp.pose.inv().to_transformation_matrix()

        self.update_render()
        ## hand_stereo
        # cam = self._cameras["hand_stereo"]
        # cam.camera.take_picture()
        # cam.camera.compute_depth()
        # trans_cam2world = cam.camera._cam_rgb.get_model_matrix()
        # cam_pc = cam.camera.get_pointcloud()
        # trans_world2ee = self.trans_cam2ee @ np.linalg.inv(trans_cam2world)
        # scene_pc = transform_points(trans_cam2world, cam_pc * np.array([1, -1, -1]))
        ## hand clean depth
        pcds = self.get_fused_pointcloud(
            cams=self.camera_modes,  # "hand_realsense", "base_kinect"
        )
        scene_pc = pcds["xyz"]
        ground_ws = (
            [0.2, 1.0],
            [-0.5, 0.5],
            [self.table_halfsize[2] * 2 - 0.0001, 0.5],
        )
        scene_pc, mask = pointcloud_filter(scene_pc, ground_ws)
        scenepcee = transform_points(trans_world2ee, scene_pc)

        # obj filter
        obj_bbdx_v = self.obj_bbdx.vertices
        trans_world2obj = self.obj_pose.inv().to_transformation_matrix()
        scene_pc_obj = transform_points(trans_world2obj, scene_pc)
        _, mask = pc_bbdx_filter(scene_pc_obj, obj_bbdx_v)
        obj_pc_filter = scene_pc[mask]
        objpcee = transform_points(trans_world2ee, obj_pc_filter)

        ## add noise to rotation & translation
        if self.points_add_noise:
            objpcee += self._episode_rng.normal(0, 0.005, (objpcee.shape[0], 3))

        return scenepcee, objpcee

    def get_state_objpoints_rt(self, action) -> OrderedDict:
        agent_qpos = self.agent.robot.get_qpos()
        cur_ee_pos, cur_ee_quat = self.tcp.pose.p, self.tcp.pose.q
        cur_ee_euler = Rotation.from_quat(np.roll(cur_ee_quat, -1)).as_euler("XYZ")
        obs = OrderedDict(
            action=action.astype(np.float32),
            gripper_pos=agent_qpos[self.agent.gripper_joint_ids].astype(np.float32),
            tcp_pose=np.hstack([cur_ee_pos, cur_ee_euler]).astype(np.float32),
        )
        _, objpcee = self.get_objpoints_rt()  # bbdx filtered obj points (ee frame)
        pc_num = objpcee.shape[0]

        objpc_num = self.obj_pc.shape[0]
        obj_pc_ee = np.zeros((objpc_num, 3))
        if 0 <= pc_num <= objpc_num:
            obj_pc_ee[:pc_num] = objpcee
        else:
            random_ids = self._episode_rng.choice(
                np.arange(pc_num), size=objpc_num, replace=False
            )
            obj_pc_ee = objpcee[random_ids]

        if self.grasps_mat is not None:
            close_grasp_pose_ee, _ = self._get_closest_grasp_pose()
        else:
            close_grasp_pose_ee = np.zeros(7)
            # close_grasp_pose_ee = np.zeros(9)

        obs.update(
            obj_pc_ee=obj_pc_ee.astype(np.float32),
            close_grasp_pose_ee=close_grasp_pose_ee.astype(np.float32),
        )
        return obs

    def _compute_near_grasps_rt(self):
        grasps_mat_ee = self.grasps_mat_ee
        grasp_ids = np.arange(grasps_mat_ee.shape[0])  # (N, 4, 4)

        if "filter" in self.grasp_select_mode:
            ## filter the grasp (<direction, grasp direction> > 60 deg)
            filter_idx = grasps_mat_ee[:, 2, 2] > np.cos(np.pi / 3)
            grasp_ids = grasp_ids[filter_idx]

        if len(grasp_ids) > self.num_grasps:
            self.grasp_ids = self._episode_rng.choice(
                grasp_ids, self.num_grasps, replace=False
            )
        else:
            self.grasp_ids = grasp_ids

        grasps_ee = np.zeros((self.num_grasps, 4, 4))
        valid_grasp_num = len(self.grasp_ids)
        grasps_ee[:valid_grasp_num] = grasps_mat_ee[self.grasp_ids]

        grasps_scores = np.zeros(self.num_grasps)
        grasps_scores[:valid_grasp_num] = self.grasps_scores[self.grasp_ids]

        return grasps_ee, grasps_scores

    def get_state_egopoints_rt(self, action):
        """
        real-time evaluation.
        """
        agent_qpos = self.agent.robot.get_qpos()
        cur_ee_pos, cur_ee_quat = self.tcp.pose.p, self.tcp.pose.q
        cur_ee_euler = Rotation.from_quat(np.roll(cur_ee_quat, -1)).as_euler("XYZ")

        obs = OrderedDict(
            gripper_pos=agent_qpos[self.agent.gripper_joint_ids].astype(np.float32),
            tcp_pose=np.hstack([cur_ee_pos, cur_ee_euler]).astype(np.float32),
            action=action.astype(np.float32),
        )

        # 아래 한줄로, 원래 region center explorer + local grasp inference 해야하는거를 대체해버림.
        grasps_ee, grasps_scores = self._compute_near_grasps_rt()

        # add grasp mask in hand_camera
        grasp_exist = np.ones(5) if len(self.grasp_ids) > 0 else np.zeros(5)

        if self.grasp_points_mode == "gauss":
            ## gaussian points
            rng = np.random.RandomState(np.random.RandomState().randint(2 ** 32))
            gripper_pts_ee = rng.normal(0.0, self.gripper_w / 3, size=(self.num_grasp_points, 3))
        elif self.grasp_points_mode == "gauss_fix":
            ## fixed gaussian points
            gripper_pts_ee = self.gripper_pts_gauss
        elif self.grasp_points_mode == "uniform":
            ## uniformly sampled points
            rng = np.random.RandomState(np.random.RandomState().randint(2 ** 32))
            gripper_pts_ee = rng.uniform(-self.gripper_w, self.gripper_w, size=(self.num_grasp_points, 3))
        elif self.grasp_points_mode == "keypoints":
            ## rigid set of keypoints
            gripper_pts_ee = self.gripper_pts_rect
        else:
            raise NotImplementedError

        R, T = grasps_ee[:, :3, :3].transpose((0, 2, 1)), grasps_ee[:, :3, 3]

        gripper_pts_diff = np.einsum("ij, kjl -> kil", gripper_pts_ee, R) + np.repeat(
            T[:, None, :], gripper_pts_ee.shape[0], axis=1
        )  # (num, k, 3)

        if self.grasps_mat is not None:
            close_grasp_pose_ee, _ = self._get_closest_grasp_pose()
        else:
            # close_grasp_pose_ee = np.zeros(7)
            close_grasp_pose_ee = np.zeros(9)

        ## self-supervised reward target
        # is_robot_static, is_obj_grasp, is_obj_static, is_obj_lift
        eval_target = np.array(self.evaluate_success())

        obs.update(
            origin_gripper_pts=gripper_pts_ee.astype(np.float32),
            grasp_exist=grasp_exist.astype(np.float32),
            gripper_pts_diff=gripper_pts_diff.astype(np.float32),
            grasps_scores=grasps_scores.astype(np.float32),
            close_grasp_pose_ee=close_grasp_pose_ee.astype(np.float32),
            eval_target=eval_target.astype(np.float32),
        )
        return obs

    def get_state_grasp9d_rt(self, action):
        """
        real-time evaluation.
        """
        agent_qpos = self.agent.robot.get_qpos()
        cur_ee_pos, cur_ee_quat = self.tcp.pose.p, self.tcp.pose.q
        cur_ee_euler = Rotation.from_quat(np.roll(cur_ee_quat, -1)).as_euler("XYZ")
        obs = OrderedDict(
            gripper_pos=agent_qpos[self.agent.gripper_joint_ids].astype(np.float32),
            tcp_pose=np.hstack([cur_ee_pos, cur_ee_euler]).astype(np.float32),
            action=action.astype(np.float32),
        )

        grasps_ee, grasps_scores = self._compute_near_grasps_rt()

        # add grasp mask in hand_camera
        grasp_exist = np.ones(5) if len(self.grasp_ids) > 0 else np.zeros(5)

        grasps_posrot_ee = np.concatenate(
            (grasps_ee[:, :3, 3], grasps_ee[:, :3, 0], grasps_ee[:, :3, 1]), axis=1
        )  # (num, 9)

        if self.grasps_mat is not None:
            close_grasp_pose_ee, _ = self._get_closest_grasp_pose()
        else:
            # close_grasp_pose_ee = np.zeros(7)
            close_grasp_pose_ee = np.zeros(9)

        obs.update(
            grasp_exist=grasp_exist.astype(np.float32),
            grasps_posrot_ee=grasps_posrot_ee.astype(np.float32),
            grasps_scores=grasps_scores.astype(np.float32),
            close_grasp_pose_ee=close_grasp_pose_ee.astype(np.float32),
        )
        return obs

    def get_done(self, info: dict, **kwargs):
        return False

    def step(self, action: Union[None, np.ndarray, Dict]):
        self.step_action(action)
        self._elapsed_steps += 1
        obs = self.get_obs(action)
        info = self.get_info(obs=obs)
        reward = self.get_reward(obs=obs, action=action, info=info)
        info.update(self._cache_info)
        done = self.get_done(obs=obs, info=info)

        # BK fix
        terminated = done
        truncated = False  # 일단 truncated는 False로 간주

        # return obs, reward, done, info
        return obs, reward, terminated, truncated, info

    def _after_simulation_step(self, sim_step):
        if not self.gen_traj_mode:
            pass
        else:
            cur_sim_steps = self._elapsed_steps * self._sim_steps_per_control + sim_step
            cur_pose = self.drive_poses[cur_sim_steps]
            cur_q = self.qs[cur_sim_steps] if self.vary_speed else self.init_q
            # cur_q = self.qs[cur_sim_steps] if self.vary_speed else self.init_q
            self.drive_base.set_pose(sapien.Pose(p=cur_pose, q=cur_q))

            ## new drive
            self.drive_base.add_force_at_point(
                -self.drive_base.get_mass() * self._scene_gravity,
                self.drive_base.pose.p,
            )
            is_agent_contact_obj, multi_impluse = self.agent.check_contact(
                self.obj, min_impulse=0.001
            )
            if is_agent_contact_obj:
                self.contact_flag = True
                self.contact_obj_pose = self.obj_pose
            if self.contact_flag:
                self.conveyor_drive.set_x_properties(stiffness=0, damping=0)
                self.conveyor_drive.set_y_properties(stiffness=0, damping=0)
                self.conveyor_drive.set_z_properties(stiffness=0, damping=0)
                self.conveyor_drive.set_slerp_properties(stiffness=0, damping=0)

    def check_robot_static(self, thresh=0.2):
        # Assume that the last two DoF is gripper
        qvel = self.agent.robot.get_qvel()[: self.num_joints]
        return np.max(np.abs(qvel)) <= thresh

    def check_obj_static(self, vel_thresh=0.05):
        vel = np.sqrt((self.obj.get_velocity() ** 2).sum(axis=-1))
        return vel < vel_thresh

    def evaluate_success(self):
        # 로봇이 정지해있나
        is_robot_static = self.check_robot_static()
        # 물체를 안정적으로 잡았나
        is_obj_grasp = self.agent.check_grasp(self.obj, max_angle=30)
        # 물체를 들어 올렸나
        obj_pose = self.obj_pose
        is_obj_lift = obj_pose.p[2] > self.goal_thresh
        # 물체가 움직이지 않고 정지 상태인가
        is_obj_static = self.check_obj_static()
        return [is_robot_static, is_obj_grasp, is_obj_static, is_obj_lift]

    def evaluate(self, **kwargs):
        is_robot_static, is_obj_grasp, is_obj_static, is_obj_lift = self.evaluate_success()
        is_success = int(is_robot_static * is_obj_grasp * is_obj_static * is_obj_lift)

        if self.obs_mode == "state_objpoints_rt":
            is_info_exist = np.any(self._get_obj_exist_mask())
        elif self.obs_mode in ["state_egopoints", "state_grasp9d"]:
            if "near" in self.grasp_select_mode:
                self._compute_near_grasps()
            is_info_exist = np.any(self._get_grasp_exist_mask())
        elif self.obs_mode in ["state_egopoints_rt", "state_grasp9d_rt"]:
            is_info_exist = np.any(self._get_grasp_exist_mask())
        else:
            is_info_exist = 0
        evaluate_info = np.array(
            [is_robot_static, is_obj_grasp, is_obj_static, is_obj_lift]
        ).astype(np.int32)
        return dict(
            is_success=is_success,
            is_info_exist=is_info_exist,
            evaluate_info=evaluate_info,
        )

    def compute_grasps_dist(self):
        if self.obs_mode == "state_objpoints_rt":
            # obj points to tcp dist
            trans_obj2world = self.obj_pose.to_transformation_matrix()
            trans_world2ee = self.tcp.pose.inv().to_transformation_matrix()
            trans_obj2ee = trans_world2ee @ trans_obj2world
            obj_pc_ee = transform_points(trans_obj2ee, self.obj_pc)
            tcp_to_obj_dist = np.linalg.norm(obj_pc_ee, axis=1)
            tcp_to_obj_dist[tcp_to_obj_dist < 1e-3] = 1  # filter zero points
            tcp_obj_dist = tcp_to_obj_dist.min()
            return [tcp_obj_dist], [1], 0

        elif self.obs_mode in ["state_egopoints", "state_grasp9d", "state_egopoints_rt", "state_grasp9d_rt"]:
            if self.obs_mode in ["state_egopoints", "state_grasp9d"]:
                # self.grasps_mat: object 좌표계에서 grasp가 위치해야 할 자세 (position of grasp from 'object frame')
                # @@@@@ 이 self.grasps_mat은 local grasp로부터 나온 값이 아니야. IV-E 2)의, GraspNet에서 나온 값이야!!!!
                
                # obj frame -> world frame
                trans_obj2world = self.obj_pose.to_transformation_matrix()
                # world frame -> ee frame
                trans_world2ee = self.tcp.pose.inv().to_transformation_matrix()
                # 먼저 제일 우항인 obj -> world를 구하고, world -> ee를 곱한다 (결국 obj -> ee frame)
                trans_obj2ee = trans_world2ee @ trans_obj2world
                # grasps_mat_ee: position of grasp from 'ee frame'
                grasps_mat_ee = np.einsum(
                    "ij, kjl -> kil", trans_obj2ee, self.grasps_mat
                )  # (N, 4, 4)
                
            elif self.obs_mode in ["state_egopoints_rt", "state_grasp9d_rt"]:
                grasps_mat_ee = self.grasps_mat_ee
                
            # dist(EE, grasp)
            translation_dist = np.linalg.norm(grasps_mat_ee[:, :3, 3], axis=1)  # (N,)
            
            # 
            quat = Rotation.from_matrix(
                grasps_mat_ee[:, :3, :3]
            ).as_quat()  # (N, 4) xyzw
            rot_dist0 = 1 - np.clip(np.abs(quat[:, 2]), a_min=0, a_max=1)
            rot_dist1 = 1 - np.clip(np.abs(quat[:, 3]), a_min=0, a_max=1)
            rotation_dist = np.minimum(rot_dist0, rot_dist1)  # (N,)

            translation_dist = np.clip(translation_dist, a_min=0.01, a_max=0.5)
            rotation_dist = np.clip(rotation_dist, a_min=0.01, a_max=0.5)
            grasp_dist = translation_dist + rotation_dist
            grasp_id = np.argmin(grasp_dist)
            return translation_dist, rotation_dist, grasp_id

    def compute_grasp_rot_dist(self):
        assert self.obs_mode in [
            "state_egopoints",
            "state_grasp9d",
            "state_egopoints_rt",
            "state_grasp9d_rt",
        ]
        if self.obs_mode in ["state_egopoints", "state_grasp9d"]:
            trans_obj2world = self.obj_pose.to_transformation_matrix()
            trans_world2ee = self.tcp.pose.inv().to_transformation_matrix()
            trans_obj2ee = trans_world2ee @ trans_obj2world
            grasps_mat_ee = np.einsum(
                "ij, kjl -> kil", trans_obj2ee, self.grasps_mat
            )  # (N, 4, 4)
        elif self.obs_mode in ["state_egopoints_rt", "state_grasp9d_rt"]:
            grasps_mat_ee = self.grasps_mat_ee
        quat = Rotation.from_matrix(grasps_mat_ee[:, :3, :3]).as_quat()  # (N, 4) xyzw
        rot_dist0 = 1 - np.clip(np.abs(quat[:, 2]), a_min=0, a_max=1)
        rot_dist1 = 1 - np.clip(np.abs(quat[:, 3]), a_min=0, a_max=1)
        rotation_dist = np.minimum(rot_dist0, rot_dist1)
        grasp_id = np.argmin(rotation_dist)
        return rotation_dist, grasp_id

    def compute_dense_reward(self, info, **kwargs):
        reward = 0.0
        approach_reward = 0.0
        grasp_reward = 0.0
        goal_reward = 0.0
        static_reward = 0.0

        translation_dist, rotation_dist, grasp_id = 0, 0, 0
        goal_dist_z = 0

        gripper_finger_contacts = self.agent.check_contact_fingers(self.obj)
        obj_pose = self.obj_pose
        obj_velangvel = np.concatenate(
            (self.obj.get_velocity(), self.obj.get_angular_velocity())
        )
        ee_pose = self.tcp.pose
        ee_velangvel = np.concatenate(
            (self.tcp.get_velocity(), self.tcp.get_angular_velocity())
        )
        arm_qvel = self.agent.robot.get_qvel()[: self.num_joints]
        arm_qacc = self.agent.robot.get_qacc()[: self.num_joints]
        arm_qf = self.agent.robot.get_qf()[: self.num_joints]

        is_robot_static, is_obj_grasp, is_obj_static, is_obj_lift = info["evaluate_info"]
        is_success = info["is_success"]

        # rw 4) grasp visibility term
        info_exist_reward = 3 if info["is_info_exist"] else 0

        # rw 3) lifting term
        if is_success:
            reward = 15.0
        else:
            # rw 1)
            # 얘는, local grasp랑 상관 없고, Sec IV-E-2)의 GraspNet의 target grasp pose (self.grasps_mat) 과의 차이를 구해서 reward를 주는 거임.
            # approach obj/grasp
            translation_dist, rotation_dist, grasp_id = self.compute_grasps_dist()
            approach_reward = 3 * (
                1 - np.tanh(5.0 * translation_dist[grasp_id])
            ) + 3 * (1 - np.tanh(5.0 * rotation_dist[grasp_id]))

            # rw 2) grasp reward
            grasp_reward = 3.0 if is_obj_grasp else 0.0

            # reach-goal reward
            if is_obj_grasp:
                # r2 3) lifting term
                ## old reward
                goal_dist_z = np.clip(
                    self.goal_thresh - obj_pose.p[2], 0, self.goal_thresh
                )  # [0, 0.2]
                goal_reward = 2 * (1 - np.tanh(5 * goal_dist_z))

                # robot static reward
                if is_obj_lift:
                    static_reward = 1 if is_obj_static and is_robot_static else 0

            reward += info_exist_reward + approach_reward + grasp_reward + static_reward + goal_reward

        reward = reward * 0.5
        info_dict = {
            "pred_target_actor_critic": self.pred_target_actor_critic,
            "trans_dist": np.array(translation_dist),
            "rot_dist": np.array(rotation_dist),
            "grasp_id": np.array(grasp_id),
            "gripper_finger_contacts": np.array(gripper_finger_contacts).astype(np.float32),
            "obj_pose": vectorize_pose(obj_pose),
            "obj_angvel": np.array(obj_velangvel),
            "ee_pose": vectorize_pose(ee_pose),
            "ee_velang": np.array(ee_velangvel),
            "arm_qvel": np.array(arm_qvel),
            "arm_qacc": np.array(arm_qacc),
            "arm_qf": np.array(arm_qf),
            "goal_dist_z": np.array(goal_dist_z),
            "info_exist_reward": float(info_exist_reward),
            "approach_reward": float(approach_reward),
            "grasp_reward": float(grasp_reward),
            "goal_reward": float(goal_reward),
            "static_reward": float(static_reward),
            "action": np.array(kwargs["action"]),
        }
        self._cache_info = info_dict
        return reward

    def render(self, mode="human", view_workspace=True, view_traj=True, view_grasps=True, view_obj_bbdx=False):
        # _view_grasps => compute near grasps 40개 후보군
        # _view_anno_grasps => 내 로봇의 gripper (빨강) + GraspNet의 저 object의 것 중 현재 내 gripper와 가장가까운 놈 (노랑)
        # _view_pred_grasp => actor의 pred_grasp (초록), critc의 pred_grasp (파랑)
        #  _view_obj_bbdx => 안들어가져서 모르겠음
        # _view_grasps => compute_near_grasps (40개 후보군)
        
        
        # 흰색 Workspace
        if view_workspace and self.gen_traj_mode in ["random2d", "bezier2d"]:
            ws = self._view_workspace()
        # object가 이동하는 경로
        if view_traj:
            traj = self._view_traj()
            
        if view_grasps and self.obs_mode in ['state_grasp9d', 'state_egopoints', 'state_egopoints_rt', 'state_grasp9d_rt']:
            if self.pred_grasp_actor_critic is not None:
                grasp_pred = self._view_pred_grasp()
            if self.grasps_mat is not None:
                grasp_anno = self._view_anno_grasps()
            grasps = self._view_grasps()
            
        if view_grasps and self.obs_mode == 'state_objpoints_rt':
            if self.grasps_mat is not None:
                grasp_anno = self._view_anno_grasps()
            if self.pred_grasp_actor_critic is not None:
                grasp_pred = self._view_pred_grasp()
                
        if view_obj_bbdx:
            obj_bbdx = self._view_obj_bbdx()
            
            
        if mode in ["human", "rgb_array"]:
            # set_actor_visibility(self.goal_site, 0.5)
            ret = super().render(mode=mode)
            # set_actor_visibility(self.goal_site, 0.0)
        else:
            ret = super().render(mode=mode)
            
            
            
        if view_workspace and self.gen_traj_mode in ["random2d", "bezier2d"]:
            self._remove_lineset(ws)
        if view_traj:
            self._remove_lineset(traj)
        if view_grasps and self.obs_mode in ['state_grasp9d', 'state_egopoints', 'state_egopoints_rt', 'state_grasp9d_rt']:
            if self.pred_grasp_actor_critic is not None:
                self._remove_lineset(grasp_pred)
            if self.grasps_mat is not None:
                self._remove_lineset(grasp_anno)
            self._remove_lineset(grasps)
        if view_grasps and self.obs_mode == 'state_objpoints_rt':
            if self.grasps_mat is not None:
                self._remove_lineset(grasp_anno)
            if self.pred_grasp_actor_critic is not None:
                self._remove_lineset(grasp_pred)
        if view_obj_bbdx:
            self._remove_lineset(obj_bbdx)
            
            
            
        return ret

    def _view_workspace(self):
        """render dynamic workspace in sapien scene"""
        assert self.dynamic_paras is not None
        x_off = self.dynamic_paras["dist"]
        half_len = self.dynamic_paras["angle"]
        end_points = np.array(
            [
                [x_off - half_len, -half_len, 0],
                [x_off - half_len, half_len, 0],
                [x_off + half_len, half_len, 0],
                [x_off + half_len, -half_len, 0],
                [x_off - half_len, -half_len, 0],
                [x_off - half_len, half_len, 0],
                [x_off + half_len, half_len, 0],
                [x_off + half_len, -half_len, 0],
            ]
        )
        if self.gen_traj_mode in ["random2d", "bezier2d"]:
            end_points[..., 2] += self.table_halfsize[2] * 2
        line_order = [
            [0, 1],
            [0, 3],
            [0, 4],
            [1, 2],
            [1, 5],
            [2, 3],
            [2, 6],
            [3, 7],
            [4, 5],
            [4, 7],
            [5, 6],
            [6, 7],
        ]
        ws_linsets = end_points[line_order].reshape(-1, 3)
        colors = [1, 1, 1, 0.5] * ws_linsets.shape[0]
        renderer_context = self._renderer._internal_context
        workspace_linesets = renderer_context.create_line_set(ws_linsets, colors)
        lineset: R.LineSetObject = (
            self._scene.renderer_scene._internal_scene.add_line_set(workspace_linesets)
        )
        return lineset

    def _view_traj(self):
        """ render 3D traj in sapien scene """
        assert self.drive_poses is not None, "you need to set gen_traj_mode first!"
        draw_poses = self.drive_poses[::50]
        draw_traj = np.tile(draw_poses, 2).reshape(-1, 3)[1:-1]
        colors = [1, 1, 0, 1] * draw_traj.shape[0]
        renderer_context = self._renderer._internal_context
        traj_linesets = renderer_context.create_line_set(draw_traj, colors)
        lineset: R.LineSetObject = (
            self._scene.renderer_scene._internal_scene.add_line_set(traj_linesets)
        )
        return lineset

    def _view_obj_bbdx(self):
        ## object bbdx
        obj_vertices = self.obj_bbdx.vertices
        trans_obj2world = self.obj_pose.to_transformation_matrix()
        obj_vertices_world = transform_points(trans_obj2world, obj_vertices)
        line_order = [[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3],
                      [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]]
        obj_linsets = obj_vertices_world[line_order].reshape(-1, 3)
        obj_colors = [0.3, 0.3, 0.3, 1] * obj_linsets.shape[0]
        renderer_context = self._renderer._internal_context
        traj_linesets = renderer_context.create_line_set(obj_linsets, obj_colors)
        lineset: R.LineSetObject = (
            self._scene.renderer_scene._internal_scene.add_line_set(traj_linesets)
        )
        return lineset

    def _view_grasps(self):
        assert self._obs_mode in ["state_egopoints", "state_grasp9d", "state_egopoints_rt", "state_grasp9d_rt"]

        trans_ee2world = self.tcp.pose.to_transformation_matrix()
        gripper_pts_world = transform_points(
            trans_ee2world, self.gripper_pts.reshape((-1, 3))
        ).reshape(self.gripper_pts.shape)  # (k, 3)

        if self.obs_mode in ["state_egopoints", "state_grasp9d"]:
            grasps_mat_ee, _ = self._compute_near_grasps()
        elif self.obs_mode in ["state_egopoints_rt", "state_grasp9d_rt"]:
            grasps_mat_ee, _ = self._compute_near_grasps_rt()

        if grasps_mat_ee.shape[0] > 0:
            R, T = grasps_mat_ee[:, :3, :3].transpose((0, 2, 1)), grasps_mat_ee[:, :3, 3]
            target_gripper_pts_ee = np.einsum(
                "ij, kjl -> kil", self.gripper_pts, R
            ) + np.repeat(T[:, None, :], self.gripper_pts.shape[0], axis=1)
            origin_shape = target_gripper_pts_ee.shape  # (N, k, 3)
            target_gripper_pts_world = transform_points(
                trans_ee2world, target_gripper_pts_ee.reshape((-1, 3))
            ).reshape(
                origin_shape
            )  # (N, k, 3)
        else:
            target_gripper_pts_world = np.zeros((1, 6, 3))

        ## eval target grasp & tcp grasp
        gripper_line_order = [0, 1, 1, 2, 2, 3, 1, 4, 4, 5]
        num_lines = len(gripper_line_order)
        draw_grasp_traj = []
        draw_grasp_traj.append(gripper_pts_world[gripper_line_order])
        colors = []
        colors.append([1, 0, 0, 1] * num_lines)

        num_grasps = grasps_mat_ee.shape[0]
        if num_grasps > 0:
            for id in range(num_grasps):
                draw_grasp_traj.append(
                    target_gripper_pts_world[id, gripper_line_order]
                )  # other grasp
            colors.append([0.5, 0, 0.5, 1] * num_grasps * num_lines)  # other grasp

        draw_grasp_traj = np.concatenate(draw_grasp_traj, axis=0)
        colors = np.concatenate(colors, axis=0)

        renderer_context = self._renderer._internal_context
        traj_linesets = renderer_context.create_line_set(draw_grasp_traj, colors)
        lineset: R.LineSetObject = (
            self._scene.renderer_scene._internal_scene.add_line_set(traj_linesets)
        )
        return lineset

    def _view_anno_grasps(self):
        assert self._obs_mode in ['state_egopoints', 'state_egopoints_rt', 'state_objpoints_rt']
        trans_ee2world = self.tcp.pose.to_transformation_matrix()
        gripper_pts_world = transform_points(
            trans_ee2world, self.gripper_pts.reshape((-1, 3))
        ).reshape(self.gripper_pts.shape)  # (k, 3)

        grasp_closest_anno_ee, min_id = self._get_closest_grasp_pose()
        grasp_closest_anno_mat_ee = np.eye(4)
        grasp_closest_anno_mat_ee[:3, 3] = grasp_closest_anno_ee[6:]
        rotz = np.cross(grasp_closest_anno_ee[:3], grasp_closest_anno_ee[3:6])
        grasp_closest_anno_mat_ee[:3, :3] = np.stack(
            (grasp_closest_anno_ee[:3], grasp_closest_anno_ee[3:6], rotz)
        ).T

        target_anno_gripper_pts_ee = transform_points(
            grasp_closest_anno_mat_ee, self.gripper_pts
        )
        target_anno_gripper_pts_world = transform_points(
            trans_ee2world, target_anno_gripper_pts_ee
        )

        ## training target grasp & tcp grasp
        gripper_line_order = [0, 1, 1, 2, 2, 3, 1, 4, 4, 5]
        num_lines = len(gripper_line_order)
        draw_grasp_traj = []
        draw_grasp_traj.append(target_anno_gripper_pts_world[gripper_line_order])
        draw_grasp_traj.append(gripper_pts_world[gripper_line_order])
        colors = []
        colors.append([1, 1, 0, 1] * num_lines)
        colors.append([1, 0, 0, 1] * num_lines)
        draw_grasp_traj = np.concatenate(draw_grasp_traj, axis=0)
        colors = np.concatenate(colors, axis=0)

        renderer_context = self._renderer._internal_context
        traj_linesets = renderer_context.create_line_set(draw_grasp_traj, colors)
        lineset: R.LineSetObject = (
            self._scene.renderer_scene._internal_scene.add_line_set(traj_linesets)
        )
        return lineset

    def _view_pred_grasp(self):
        assert self.pred_grasp_actor_critic is not None, self.pred_grasp_actor_critic
        trans_ee2world = self.tcp.pose.to_transformation_matrix()

        pred_gripper_pts_world_actor_critic = []
        for i in range(2):
            pred_mat_ee = np.eye(4)
            pred_mat_ee[:3, 3] = self.pred_grasp_actor_critic[i, 6:]
            rotz = np.cross(self.pred_grasp_actor_critic[i, :3], self.pred_grasp_actor_critic[i, 3:6])
            pred_mat_ee[:3, :3] = np.stack(
                (self.pred_grasp_actor_critic[i, :3], self.pred_grasp_actor_critic[i, 3:6], rotz)
            ).T
            pred_gripper_pts_ee = transform_points(
                pred_mat_ee, self.gripper_pts
            )
            pred_gripper_pts_world = transform_points(
                trans_ee2world, pred_gripper_pts_ee
            )
            pred_gripper_pts_world_actor_critic.append(pred_gripper_pts_world)

        ## training target grasp & tcp grasp
        gripper_line_order = [0, 1, 1, 2, 2, 3, 1, 4, 4, 5]
        num_lines = len(gripper_line_order)
        draw_grasp_traj = []
        draw_grasp_traj.append(pred_gripper_pts_world_actor_critic[0][gripper_line_order])
        draw_grasp_traj.append(pred_gripper_pts_world_actor_critic[1][gripper_line_order])
        colors = []
        colors.append([0, 1, 0, 1] * num_lines)
        colors.append([0, 0, 1, 1] * num_lines)

        renderer_context = self._renderer._internal_context
        traj_linesets = renderer_context.create_line_set(draw_grasp_traj, colors)
        lineset: R.LineSetObject = (
            self._scene.renderer_scene._internal_scene.add_line_set(traj_linesets)
        )
        return lineset

    def _remove_lineset(self, lineset):
        self._scene.renderer_scene._internal_scene.remove_node(lineset)

    def _register_render_cameras(self):
        if self.robot_uid == "panda":
            pose = look_at([0.4, 0.4, 0.8], [0.0, 0.0, 0.4])
        else:
            pose = look_at(
                [0.8 + self.robot_x_offset, 0.5, 1.0], [self.robot_x_offset, 0, 0.5]
            )
        return CameraConfig("render_camera", pose.p, pose.q, 512, 512, 1, 0.01, 10)

    def _setup_viewer(self):
        super()._setup_viewer()
        self._viewer.set_camera_xyz(0.8 + self.robot_x_offset, 0, 1.0)
        self._viewer.set_camera_rpy(0, -0.5, 3.14)

    def _get_obs_agent(self):
        obs = self.agent.get_proprioception()
        obs["base_pose"] = vectorize_pose(self.agent.robot.pose)
        return obs

    def get_state(self) -> np.ndarray:
        state = super().get_state()
        return np.hstack([state, self.goal_pos])

    def set_state(self, state):
        self.goal_pos = state[-3:]
        super().set_state(state[:-3])


# ---------------------------------------------------------------------------- #
# YCB
# ---------------------------------------------------------------------------- #
def build_actor_ycb(
    model_id: str,
    scene: sapien.Scene,
    scale: float = 1.0,
    physical_material: sapien.PhysicalMaterial = None,
    density=1000,
    root_dir=ASSET_DIR / "mani_skill2_ycb",
):
    builder = scene.create_actor_builder()
    model_dir = Path(root_dir) / "models" / model_id

    collision_file = str(model_dir / "collision_coacd.obj")
    builder.add_multiple_collisions_from_file(
        filename=collision_file,
        scale=[scale] * 3,
        material=physical_material,
        density=density,
    )

    visual_file = str(model_dir / "textured.obj")
    builder.add_visual_from_file(filename=visual_file, scale=[scale] * 3)

    actor = builder.build()
    return actor


@register_env("PickSingleYCB-v0", max_episode_steps=100)
class PickSingleYCBEnv(PickSingleEnv):
    DEFAULT_ASSET_ROOT = "{ASSET_DIR}/mani_skill2_ycb"
    DEFAULT_MODEL_JSON = "info_pick_v1.json"
    DEFAULT_GRASP_JSON = "info_graspnet_v1.json"
    DEFAULT_LOCALGRASP_JSON = "info_localgrasp_v3.json"

    def __init__(
        self,
        **kwargs,
    ):
        
        # BK add
        self.render_mode = None
        # self.render_mode = kwargs.get("render_mode", None)

        asset_root = Path(format_path(self.DEFAULT_ASSET_ROOT))

        # 미리 github에 나와있는 설명처럼 near_grasp들 미리 계산한거 불러오자
        grasp_json = asset_root / format_path(self.DEFAULT_GRASP_JSON)
        if not grasp_json.exists():
            raise FileNotFoundError(
                f"{grasp_json} is not found."
                "Please generate the corresponding assets using HGGD. "
            )
        grasp_db: Dict[str, Dict] = load_json(grasp_json)

        lg_json = asset_root / format_path(self.DEFAULT_LOCALGRASP_JSON)
        if not lg_json.exists():
            raise FileNotFoundError(
                f"{lg_json} is not found."
                "Please generate the corresponding assets using LocalGrasp. "
            )
        lg_db: Dict[str, Dict] = load_json(lg_json)

        all_grasps = OrderedDict()
        all_lg_grasps = OrderedDict()
        for model_id in kwargs["model_ids"]:
            grasps = np.array(grasp_db[model_id]["transformations"])  # (N, 7)
            all_grasps[model_id] = grasps
            all_lg_grasps[model_id] = lg_db[model_id]["grasp"]  # (72, dict)
        self.all_grasps = all_grasps
        self.all_lg_grasps = all_lg_grasps

        super().__init__(**kwargs)

    def _load_model(self):
        density = self.model_db[self.model_id].get("density", 1000)
        phy_mat = self._scene.create_physical_material(
            static_friction=1.0, dynamic_friction=1.0, restitution=0.0
        )
        self.obj = build_actor_ycb(
            self.model_id,
            self._scene,
            scale=self.model_scale,
            physical_material=phy_mat,
            density=density,
            root_dir=self.asset_root,
        )
        self.obj.name = self.model_id
        obj_mesh = get_actor_mesh(self.obj, to_world_frame=False)
        self.obj_pc = obj_mesh.sample(256)  # obj frame
        self.obj_bbdx = obj_mesh.bounding_box
        self.obj_aabb_halfsize = self.obj_bbdx.extents / 2

        cur_grasp_poses = self.all_grasps[self.model_id]  # (N, 7)
        grasp_mats = np.repeat(np.eye(4)[None], cur_grasp_poses.shape[0], 0)
        grasp_mats[:, :3, 3] = cur_grasp_poses[:, :3]
        grasp_mats[:, :3, :3] = Rotation.from_quat(cur_grasp_poses[:, 3:]).as_matrix()
        self.grasps_mat = grasp_mats  # (N, 4, 4)

        if self.obs_mode in ["state_egopoints", "state_grasp9d", "state_grasp_obj_points"]:
            self.lg_grasps_dict = self.all_lg_grasps[self.model_id]  # (72, dict)
            grasp_views = len(self.lg_grasps_dict)
            if self.grasp_select_mode in ["random", "angle_filter"]:
                lg_grasps_poses = []
                lg_grasps_scores = []
                for grasp_view_id in range(grasp_views):
                    if self.lg_grasps_dict[grasp_view_id] is not None:
                        transformations = self.lg_grasps_dict[grasp_view_id]['transformations']
                        lg_grasps_poses.extend(transformations)
                        scores = self.lg_grasps_dict[grasp_view_id]['scores']
                        lg_grasps_scores.extend(scores)
                lg_grasps_poses = np.array(lg_grasps_poses)
                grasp_mats = np.repeat(np.eye(4)[None], lg_grasps_poses.shape[0], 0)
                grasp_mats[:, :3, 3] = lg_grasps_poses[:, :3]
                grasp_mats[:, :3, :3] = Rotation.from_quat(lg_grasps_poses[:, 3:]).as_matrix()
                self.lg_grasps_mat = grasp_mats  # (M, 4, 4)
                self.lg_grasps_score = np.array(lg_grasps_scores)

    def _get_init_z(self):
        return self.obj_aabb_halfsize[2]


@register_env("PickSingleYCB-v1", max_episode_steps=100)
class PickSingleYCBEnv_v1(PickSingleEnv):
    DEFAULT_ASSET_ROOT = "{ASSET_DIR}/mani_skill2_ycb"
    DEFAULT_MODEL_JSON = "info_pick_v1.json"
    DEFAULT_GRASP_JSON = "info_graspnet_v1.json"
    DEFAULT_LOCALGRASP_JSON = "info_localgrasp_v0.json"

    def __init__(
        self,
        **kwargs,
    ):
        self.use_stereo = False
        super().__init__(**kwargs)

    def set_stereo_mode(self, is_stereo=False):
        self.use_stereo = is_stereo

    def _load_actors(self):
        self._add_ground(render=self.bg_name is None)
        self._load_model()
        self.obj.set_damping(0.1, 0.1)

    def _load_model(self):
        density = self.model_db[self.model_id].get("density", 1000)
        phy_mat = self._scene.create_physical_material(
            static_friction=1.0, dynamic_friction=1.0, restitution=0.0
        )
        self.obj = build_actor_ycb(
            self.model_id,
            self._scene,
            scale=self.model_scale,
            physical_material=phy_mat,
            density=density,
            root_dir=self.asset_root,
        )
        self.obj.name = self.model_id
        obj_mesh = get_actor_mesh(self.obj, to_world_frame=False)
        self.obj_pc = obj_mesh.sample(256)  # obj frame
        self.obj_bbdx = obj_mesh.bounding_box
        self.obj_aabb_halfsize = self.obj_bbdx.extents / 2

    def _initialize_task(self, max_trials=100):
        pass

    def _initialize_agent(self):
        """Initialize the (joint) poses of agent(robot)."""
        qpos = np.array(
            [-1.27 , -0.9, -1.6, -1.7, 1.4, -2.7, 0, 0]  # top-down view
        )
        qpos[:-2] += self._episode_rng.normal(
            0, self.robot_init_qpos_noise, len(qpos) - 2
        )
        self.agent.reset(qpos)
        self.agent.robot.set_pose(Pose(p=[-10,0,0]))
        # get cam2ee transform
        handcam_joint = get_entity_by_name(self.agent.robot.get_joints(), "realsense_hand_joint")
        self.trans_cam2ee = handcam_joint.get_pose_in_parent().to_transformation_matrix()

    def _initialize_actors(self):
        obj_p = np.array([0, 0, self._get_init_z()])
        self.obj.set_pose(Pose(obj_p))
        self.obj_init_pos = self.obj.pose.p

    def get_state_objpoints_rt(self, action) -> OrderedDict:
        cam = self._cameras["data_cam"]
        self.update_render()
        if self.use_stereo:
            cam.camera.take_picture()
            cam.camera.compute_depth()
            trans_cam2world = cam.camera._cam_rgb.get_model_matrix()
            cam_pc = cam.camera.get_pointcloud()
        else:
            cam.take_picture()
            trans_cam2world = cam.camera.get_model_matrix()
            cam_pc = cam.get_camera_pcd(rgb=False, visual_seg=False, actor_seg=False)['xyz']
        trans_world2ee = self.trans_cam2ee @ np.linalg.inv(trans_cam2world)

        scene_pc = transform_points(trans_cam2world, cam_pc)

        ground_ws = (
            [-0.5, 0.5],
            [-0.5, 0.5],
            [-0.0001, 0.5],
        )
        scene_pc, mask = pointcloud_filter(scene_pc, ground_ws)
        scenepcee = transform_points(trans_world2ee, scene_pc)

        # obj filter
        obj_bbdx_v = self.obj_bbdx.vertices
        trans_world2obj = self.obj_pose.inv().to_transformation_matrix()
        scene_pc_obj = transform_points(trans_world2obj, scene_pc)
        _, mask = pc_bbdx_filter(scene_pc_obj, obj_bbdx_v)
        objpcee = scenepcee[mask]
        # scenepcee = scenepcee[~mask]

        obs = OrderedDict(
            obj_pc_ee=objpcee.astype(np.float32),
            scene_pc_ee=scenepcee.astype(np.float32),
            # scene_pc_world=scene_pc.astype(np.float32),
            # close_grasp_pose_ee=close_grasp_pose_ee.astype(np.float32),
        )
        return obs

    def _get_init_z(self):
        return self.obj_aabb_halfsize[2]


@register_env("PickSingleYCB-v3", max_episode_steps=100)
class PickSingleYCBEnv_v3(PickSingleEnv):
    DEFAULT_ASSET_ROOT = "{ASSET_DIR}/mani_skill2_ycb"
    DEFAULT_MODEL_JSON = "info_pick_eval.json"

    def _load_model(self):
        density = self.model_db[self.model_id].get("density", 1000)
        phy_mat = self._scene.create_physical_material(
            static_friction=1.0, dynamic_friction=1.0, restitution=0.0
        )
        self.obj = build_actor_ycb(
            self.model_id,
            self._scene,
            scale=self.model_scale,
            physical_material=phy_mat,
            density=density,
            root_dir=self.asset_root,
        )
        self.obj.name = self.model_id
        obj_mesh = get_actor_mesh(self.obj, to_world_frame=False)
        self.obj_pc = obj_mesh.sample(256)  # obj frame
        self.obj_bbdx = obj_mesh.bounding_box
        self.obj_aabb_halfsize = self.obj_bbdx.extents / 2

    def _get_init_z(self):
        return self.obj_aabb_halfsize[2]

    def compute_dense_reward(self, info, **kwargs):
        return 0


@register_env("PickSingleACRONYM-v0", max_episode_steps=100)
class PickSingleACRONYMEnv(PickSingleEnv):
    DEFAULT_ASSET_ROOT = "{ASSET_DIR}/mani_skill2_acronym"
    DEFAULT_MODEL_JSON = "info_pick_v0.json"
    DEFAULT_GRASP_JSON = "info_grasp_v2.json"

    def _load_model(self):
        mat = self._renderer.create_material()
        color = self._episode_rng.uniform(0.2, 0.8, 3)
        color = np.hstack([color, 1.0])
        mat.set_base_color(color)
        mat.metallic = 0.0
        mat.roughness = 0.1
        phy_mat = self._scene.create_physical_material(
            static_friction=1, dynamic_friction=1, restitution=0.0
        )

        builder = self._scene.create_actor_builder()
        model_hash_id = self.model_db[self.model_id]["id"]
        origin_scale, self.model_scale = self.model_db[self.model_id]["scale"]
        model_dir = (
            Path(self.asset_root) / f"{self.model_id}_{model_hash_id}_{origin_scale}"
        )

        collision_file = str(model_dir / f"{model_hash_id}_coacd_norm.obj")
        builder.add_multiple_collisions_from_file(
            filename=collision_file,
            scale=[self.model_scale] * 3,
            material=phy_mat,
            density=1e3,
        )

        visual_file = str(model_dir / f"{model_hash_id}_coacd_norm.obj")
        builder.add_visual_from_file(
            filename=visual_file, scale=[self.model_scale] * 3, material=mat
        )

        self.obj = builder.build()
        self.obj.name = self.model_id

        obj_mesh = get_actor_mesh(self.obj, to_world_frame=True)
        # obj_pc = obj_mesh.sample(2048)  # obj frame
        obj_mesh_o3d = obj_mesh.as_open3d
        obj_pc = np.asarray(
            obj_mesh_o3d.sample_points_uniformly(256).points
        )  # obj frame
        # import open3d as o3d
        # mesh = o3d.io.read_triangle_mesh(visual_file)
        # obj_pc = np.asarray(mesh.sample_points_uniformly(2048).points)
        self.obj_pc = obj_pc
        self.obj_bbdx = obj_mesh.bounding_box
        self.obj_aabb_halfsize = self.obj_bbdx.extents / 2

    def compute_dense_reward(self, info, **kwargs):
        return 0

    def _load_actors(self):
        self._add_ground(render=self.bg_name is None)
        self._load_model()
        self.obj.set_damping(0.1, 0.1)
        half_height = 0.005
        self.box_halfsize = (self.obj_aabb_halfsize[:2] + 0.03).tolist() + [half_height]
        phy_mat = self._scene.create_physical_material(
            static_friction=1, dynamic_friction=1, restitution=0.0
        )
        self.drive_base = self._build_box(
            pose=Pose([self.robot_x_offset, 0, self.box_halfsize[2]]),
            phy_mat=phy_mat,
            half_size=self.box_halfsize,
            density=1e6,
            color=(0, 0, 0),
            name="drive_base",
            hide_visual=False,
        )

        drive_base_pose = self.drive_base.pose
        drive_base_pose.set_p(-drive_base_pose.p)
        self.conveyor_drive = self._scene.create_drive(
            None, Pose(), self.drive_base, Pose()
        )
        self.conveyor_drive.lock_motion(False, False, False, False, False, False)

    def _get_init_z(self):
        return self.obj_aabb_halfsize[2]

    def _get_obj_init_xyz(self, drive_base_p):
        obj_p = drive_base_p + np.array(
            [0, 0, self.obj_aabb_halfsize[2] + self.box_halfsize[2]]
        )
        return obj_p

    def gen_scene_pcd(self, num_points: List) -> np.ndarray:
        """Generate scene point cloud for motion planning, excluding the robot"""
        scene_obj_pcd = []
        scene_ground_pcd = []
        actor_obj_mesh = None
        actor_ground_mesh = None
        for actor in self._scene.get_all_actors():
            if actor == self.obj:
                actor_obj_mesh = merge_meshes(get_actor_visual_meshes(actor))
            else:
                actor_ground_mesh = merge_meshes(get_actor_meshes(actor))
            if actor_obj_mesh:
                actor_obj_mesh.apply_transform(
                    actor.get_pose().to_transformation_matrix()
                )
                scene_obj_pcd.append(actor_obj_mesh.sample(num_points[0]))
            elif actor_ground_mesh:
                actor_ground_mesh.apply_transform(
                    actor.get_pose().to_transformation_matrix()
                )
                scene_ground_pcd.append(actor_ground_mesh.sample(num_points[1]))

        return np.concatenate(scene_obj_pcd, axis=0), np.concatenate(scene_ground_pcd, axis=0)


@register_env("PickSingleGraspnet-v0", max_episode_steps=100)
class PickSingleGraspnetEnv(PickSingleEnv):
    DEFAULT_ASSET_ROOT = "{ASSET_DIR}/mani_skill2_graspnet"
    DEFAULT_MODEL_JSON = "info_pick_v0.json"
    DEFAULT_GRASP_JSON = "info_grasp_v0.json"

    def _get_init_z(self):
        return self.obj_aabb_halfsize[2]

    def _get_obj_init_xyz(self, drive_base_p):
        obj_p = drive_base_p + np.array(
            [0, 0, self.obj_aabb_halfsize[2] + self.box_halfsize[2]]
        )
        return obj_p

    def _load_model(self):
        mat = self._renderer.create_material()
        color = self._episode_rng.uniform(0.2, 0.8, 3)
        color = np.hstack([color, 1.0])
        mat.set_base_color(color)
        mat.metallic = 0.0
        mat.roughness = 0.1
        phy_mat = self._scene.create_physical_material(
            static_friction=1, dynamic_friction=1, restitution=0.0
        )

        builder = self._scene.create_actor_builder()
        model_name = self.model_db[self.model_id]["name"]
        self.model_scale = self.model_db[self.model_id]["scale"]
        model_dir = Path(self.asset_root) / f"{self.model_id}"

        collision_file = str(model_dir / f"textured_0_coacd.obj")
        builder.add_multiple_collisions_from_file(
            filename=collision_file,
            scale=[self.model_scale] * 3,
            material=phy_mat,
            density=1e3,
        )

        visual_file = str(model_dir / f"textured_0_coacd.obj")
        builder.add_visual_from_file(
            filename=visual_file, scale=[self.model_scale] * 3, material=mat
        )

        self.obj = builder.build()
        self.obj.name = model_name

        obj_mesh = get_actor_mesh(self.obj, to_world_frame=True)
        # obj_pc = obj_mesh.sample(2048)  # obj frame
        obj_mesh_o3d = obj_mesh.as_open3d
        obj_pc = np.asarray(
            obj_mesh_o3d.sample_points_uniformly(256).points
        )  # obj frame
        # import open3d as o3d
        # mesh = o3d.io.read_triangle_mesh(visual_file)
        # obj_pc = np.asarray(mesh.sample_points_uniformly(2048).points)
        self.obj_pc = obj_pc
        self.obj_bbdx = obj_mesh.bounding_box
        self.obj_aabb_halfsize = self.obj_bbdx.extents / 2

    def _load_actors(self):
        self._add_ground(render=self.bg_name is None)
        self._load_model()
        self.obj.set_damping(0.1, 0.1)
        half_height = 0.005
        self.box_halfsize = (self.obj_aabb_halfsize[:2] + 0.03).tolist() + [half_height]
        phy_mat = self._scene.create_physical_material(
            static_friction=1, dynamic_friction=1, restitution=0.0
        )
        self.drive_base = self._build_box(
            pose=Pose([self.robot_x_offset, 0, self.box_halfsize[2]]),
            phy_mat=phy_mat,
            half_size=self.box_halfsize,
            density=1e6,
            color=(0, 0, 0),
            name="drive_base",
            hide_visual=False,
        )

        drive_base_pose = self.drive_base.pose
        drive_base_pose.set_p(-drive_base_pose.p)
        self.conveyor_drive = self._scene.create_drive(
            None, Pose(), self.drive_base, Pose()
        )
        self.conveyor_drive.lock_motion(False, False, False, False, False, False)

    def compute_dense_reward(self, info, **kwargs):
        return 0


# ---------------------------------------------------------------------------- #
# EGAD
# ---------------------------------------------------------------------------- #
def build_actor_egad(
    model_id: str,
    scene: sapien.Scene,
    scale: float = 1.0,
    physical_material: sapien.PhysicalMaterial = None,
    density=100,
    render_material: sapien.RenderMaterial = None,
    root_dir=ASSET_DIR / "mani_skill2_egad",
):
    builder = scene.create_actor_builder()
    # A heuristic way to infer split
    # split = "train" if "_" in model_id else "eval"
    split = "eval"

    collision_file = Path(root_dir) / f"egad_{split}_set_coacd" / f"{model_id}.obj"
    builder.add_multiple_collisions_from_file(
        filename=str(collision_file),
        scale=[scale] * 3,
        material=physical_material,
        density=density,
    )

    visual_file = Path(root_dir) / f"egad_{split}_set" / f"{model_id}.obj"
    builder.add_visual_from_file(
        filename=str(visual_file), scale=[scale] * 3, material=render_material
    )

    actor = builder.build()
    return actor


@register_env("PickSingleEGAD-v0", max_episode_steps=100)
class PickSingleEGADEnv(PickSingleEnv):
    DEFAULT_ASSET_ROOT = "{ASSET_DIR}/mani_skill2_egad"
    DEFAULT_MODEL_JSON = "info_pick_eval_v0.json"

    def __init__(
        self,
        robot="panda",
        robot_init_qpos_noise=0.02,
        asset_root: str = None,
        model_json: str = None,
        num_grasps: int = 10,
        model_ids: List[str] = (),
        obj_init_rot_z: bool = True,
        obj_init_rot: float = 0.0,
        goal_thresh: float = 0.2,
        goal_pos: List[float] = [0.5, 0.0, 0.3],
        robot_x_offset: float = 0.56,
        gen_traj_mode: str = None,
        **kwargs,
    ):
        if asset_root is None:
            asset_root = self.DEFAULT_ASSET_ROOT
        self.asset_root = Path(format_path(asset_root))

        if model_json is None:
            model_json = self.DEFAULT_MODEL_JSON
        model_json = self.asset_root / format_path(model_json)

        if not model_json.exists():
            raise FileNotFoundError(
                f"{model_json} is not found."
                "Please download the corresponding assets:"
                "`python -m gap_rl.utils.download_asset ${ENV_ID}`."
            )
        self.model_db: Dict[str, Dict] = load_json(model_json)

        if isinstance(model_ids, str):
            model_ids = [model_ids]
        if len(model_ids) == 0:
            model_ids = sorted(self.model_db.keys())
        assert len(model_ids) > 0, model_json
        self.model_ids = model_ids

        self.num_grasps = num_grasps
        self.grasps_mat_ee = np.zeros((num_grasps, 4, 4))

        self.model_id = model_ids[0]
        self.model_scale = None
        self.model_bbox_size = None

        self.obj_init_rot_z = obj_init_rot_z
        self.obj_init_rot = obj_init_rot
        self.goal_thresh = goal_thresh
        self.goal_pos = goal_pos
        self.robot_x_offset = robot_x_offset
        self.gen_traj_mode = gen_traj_mode
        self.vary_speed = kwargs.pop("vary_speed", False)

        self.contact_obj_pose = Pose()
        self.grasps_mat = None
        self.grasps_scores = None
        self.grasp_ids = None
        self.grasps_ids_sort = None

        self.robot_uid = robot
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.contact_flag = False

        self._cache_info = {}

        self._check_assets()
        BaseEnv.__init__(self, **kwargs)

    def _check_assets(self):
        splits = set()
        for model_id in self.model_ids:
            split = "train" if "_" in model_id else "eval"
            splits.add(split)

        for split in splits:
            collision_dir = self.asset_root / f"egad_{split}_set_coacd"
            visual_dir = self.asset_root / f"egad_{split}_set"
            if not (collision_dir.exists() and visual_dir.exists()):
                raise FileNotFoundError(
                    f"{collision_dir} or {visual_dir} is not found. "
                    "Please download (ManiSkill2) EGAD models:"
                    "`python -m gap_rl.utils.download_asset egad`."
                )

    def _load_model(self):
        mat = self._renderer.create_material()
        color = self._episode_rng.uniform(0.2, 0.8, 3)
        color = np.hstack([color, 1.0])
        mat.set_base_color(color)
        mat.metallic = 0.0
        mat.roughness = 0.1

        self.obj = build_actor_egad(
            self.model_id,
            self._scene,
            scale=self.model_scale,
            render_material=mat,
            density=1e3,
            root_dir=self.asset_root,
        )
        self.obj.name = self.model_id
        obj_mesh = get_actor_mesh(self.obj, to_world_frame=False)
        obj_pc = obj_mesh.sample(256)  # obj frame
        self.obj_aabb_halfsize = obj_mesh.bounding_box.extents / 2
        self.obj_pc = obj_pc

    def _load_actors(self):
        self._add_ground(render=self.bg_name is None)
        self._load_model()
        self.obj.set_damping(0.1, 0.1)
        half_height = 0.005
        self.box_halfsize = (self.obj_aabb_halfsize[:2] + 0.03).tolist() + [half_height]
        phy_mat = self._scene.create_physical_material(
            static_friction=1, dynamic_friction=1, restitution=0.0
        )
        self.drive_base = self._build_box(
            pose=Pose([self.robot_x_offset, 0, self.box_halfsize[2]]),
            phy_mat=phy_mat,
            half_size=self.box_halfsize,
            density=1e6,
            color=(0, 0, 0),
            name="drive_base",
            hide_visual=False,
        )

        drive_base_pose = self.drive_base.pose
        drive_base_pose.set_p(-drive_base_pose.p)
        self.conveyor_drive = self._scene.create_drive(
            None, Pose(), self.drive_base, Pose()
        )
        self.conveyor_drive.lock_motion(False, False, False, False, False, False)

    def _get_init_z(self):
        return self.obj_aabb_halfsize[2]

    def _get_obj_init_xyz(self, drive_base_p):
        obj_p = drive_base_p - np.append(self.obj_aabb_halfsize[:2], 0)
        return obj_p

    def compute_dense_reward(self, info, **kwargs):
        return 0
