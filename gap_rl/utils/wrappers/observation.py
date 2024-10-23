from collections import OrderedDict, deque
from copy import deepcopy
from typing import Sequence

import gym
import numpy as np
from gap_rl.utils.common import (
    flatten_dict_keys,
    flatten_dict_space_keys,
    merge_dicts,
)
from gym import spaces
from gym.wrappers import LazyFrames


class RGBDObservationWrapper(gym.ObservationWrapper):
    """Map raw textures (Color and Position) to rgb and depth."""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = deepcopy(env.observation_space)
        self.update_observation_space(self.observation_space)

    @staticmethod
    def update_observation_space(space: spaces.Dict):
        # Update image observation space
        image_space: spaces.Dict = space.spaces["image"]
        for cam_uid in image_space:
            ori_cam_space = image_space[cam_uid]
            new_cam_space = OrderedDict()
            for key in ori_cam_space:
                if key == "Color":
                    height, width = ori_cam_space[key].shape[:2]
                    new_cam_space["rgb"] = spaces.Box(
                        low=0, high=255, shape=(height, width, 3), dtype=np.uint8
                    )
                elif key == "Position":
                    height, width = ori_cam_space[key].shape[:2]
                    new_cam_space["depth"] = spaces.Box(
                        low=0, high=np.inf, shape=(height, width, 1), dtype=np.float32
                    )
                else:
                    new_cam_space[key] = ori_cam_space[key]
            image_space.spaces[cam_uid] = spaces.Dict(new_cam_space)

    def observation(self, observation: dict):
        image_obs = observation["image"]
        for cam_uid, ori_images in image_obs.items():
            new_images = OrderedDict()
            for key in ori_images:
                if key == "Color":
                    rgb = ori_images[key][..., :3]  # [H, W, 4]
                    rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
                    new_images["rgb"] = rgb  # [H, W, 4]
                elif key == "Position":
                    depth = -ori_images[key][..., [2]]  # [H, W, 1]
                    new_images["depth"] = depth
                else:
                    new_images[key] = ori_images[key]
            image_obs[cam_uid] = new_images
        return observation


def merge_dict_spaces(dict_spaces: Sequence[spaces.Dict]):
    reverse_spaces = merge_dicts([x.spaces for x in dict_spaces])
    for key in reverse_spaces:
        low, high = [], []
        for x in reverse_spaces[key]:
            assert isinstance(x, spaces.Box), type(x)
            low.append(x.low)
            high.append(x.high)
        low = np.concatenate(low)
        high = np.concatenate(high)
        new_space = spaces.Box(low=low, high=high, dtype=low.dtype)
        reverse_spaces[key] = new_space
    return spaces.Dict(OrderedDict(reverse_spaces))


class PointCloudObservationWrapper(gym.ObservationWrapper):
    """Convert Position textures to world-space point cloud."""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = deepcopy(env.observation_space)
        self.update_observation_space(self.observation_space)
        self._buffer = {}

    @staticmethod
    def update_observation_space(space: spaces.Dict):
        # Replace image observation spaces with point cloud ones
        image_space: spaces.Dict = space.spaces.pop("image")
        space.spaces.pop("camera_param")
        pcd_space = OrderedDict()

        for cam_uid in image_space:
            cam_image_space = image_space[cam_uid]
            cam_pcd_space = OrderedDict()

            h, w = cam_image_space["Position"].shape[:2]
            cam_pcd_space["xyzw"] = spaces.Box(
                low=-np.inf, high=np.inf, shape=(h * w, 4), dtype=np.float32
            )

            # Extra keys
            if "Color" in cam_image_space.spaces:
                cam_pcd_space["rgb"] = spaces.Box(
                    low=0, high=255, shape=(h * w, 3), dtype=np.uint8
                )
            if "Segmentation" in cam_image_space.spaces:
                cam_pcd_space["Segmentation"] = spaces.Box(
                    low=0, high=(2 ** 32 - 1), shape=(h * w, 4), dtype=np.uint32
                )

            pcd_space[cam_uid] = spaces.Dict(cam_pcd_space)

        pcd_space = merge_dict_spaces(pcd_space.values())
        space.spaces["pointcloud"] = pcd_space

    def observation(self, observation: dict):
        image_obs = observation.pop("image")
        camera_params = observation.pop("camera_param")
        pointcloud_obs = OrderedDict()

        for cam_uid, images in image_obs.items():
            cam_pcd = {}

            # Each pixel is (x, y, z, z_buffer_depth) in OpenGL camera space
            position = images["Position"]
            # position[..., 3] = position[..., 3] < 1
            position[..., 3] = position[..., 2] < 0

            # Convert to world space
            cam2world = camera_params[cam_uid]["cam2world_gl"]
            xyzw = position.reshape(-1, 4) @ cam2world.T
            cam_pcd["xyzw"] = xyzw

            # Extra keys
            if "Color" in images:
                rgb = images["Color"][..., :3]
                rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
                cam_pcd["rgb"] = rgb.reshape(-1, 3)
            if "Segmentation" in images:
                cam_pcd["Segmentation"] = images["Segmentation"].reshape(-1, 4)

            pointcloud_obs[cam_uid] = cam_pcd

        pointcloud_obs = merge_dicts(pointcloud_obs.values())
        for key, value in pointcloud_obs.items():
            buffer = self._buffer.get(key, None)
            pointcloud_obs[key] = np.concatenate(value, out=buffer)
            self._buffer[key] = pointcloud_obs[key]

        observation["pointcloud"] = pointcloud_obs
        return observation


class RobotSegmentationObservationWrapper(gym.ObservationWrapper):
    """Add a binary mask for robot links."""

    def __init__(self, env, replace=True):
        super().__init__(env)
        self.observation_space = deepcopy(env.observation_space)
        self.init_observation_space(self.observation_space, replace=replace)
        self.replace = replace
        # Cache robot link ids
        self.robot_link_ids = self.env.robot_link_ids

    @staticmethod
    def init_observation_space(space: spaces.Dict, replace: bool):
        # Update image observation spaces
        if "image" in space.spaces:
            image_space = space["image"]
            for cam_uid in image_space:
                cam_space = image_space[cam_uid]
                if "Segmentation" not in cam_space.spaces:
                    continue
                height, width = cam_space["Segmentation"].shape[:2]
                new_space = spaces.Box(
                    low=0, high=1, shape=(height, width, 1), dtype="bool"
                )
                if replace:
                    cam_space.spaces.pop("Segmentation")
                cam_space.spaces["robot_seg"] = new_space

        # Update pointcloud observation spaces
        if "pointcloud" in space.spaces:
            pcd_space = space["pointcloud"]
            if "Segmentation" in pcd_space.spaces:
                n = pcd_space["Segmentation"].shape[0]
                new_space = spaces.Box(low=0, high=1, shape=(n, 1), dtype="bool")
                if replace:
                    pcd_space.spaces.pop("Segmentation")
                pcd_space.spaces["robot_seg"] = new_space

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        self.robot_link_ids = self.env.robot_link_ids
        return self.observation(observation)

    def observation_image(self, observation: dict):
        image_obs = observation["image"]
        for cam_images in image_obs.values():
            if "Segmentation" not in cam_images:
                continue
            seg = cam_images["Segmentation"]
            robot_seg = np.isin(seg[..., 1:2], self.robot_link_ids)
            if self.replace:
                cam_images.pop("Segmentation")
            cam_images["robot_seg"] = robot_seg
        return observation

    def observation_pointcloud(self, observation: dict):
        pointcloud_obs = observation["pointcloud"]
        if "Segmentation" not in pointcloud_obs:
            return observation
        seg = pointcloud_obs["Segmentation"]
        robot_seg = np.isin(seg[..., 1:2], self.robot_link_ids)
        if self.replace:
            pointcloud_obs.pop("Segmentation")
        pointcloud_obs["robot_seg"] = robot_seg
        return observation

    def observation(self, observation: dict):
        if "image" in observation:
            observation = self.observation_image(observation)
        if "pointcloud" in observation:
            observation = self.observation_pointcloud(observation)
        return observation


class FlattenObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env) -> None:
        super().__init__(env)
        self.observation_space = flatten_dict_space_keys(self.observation_space)

    def observation(self, observation):
        return flatten_dict_keys(observation)


class StackObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env) -> None:
        super().__init__(env)
        new_space: spaces.Dict = env.observation_space.spaces
        for key in new_space.keys():
            if key == "rgb":
                new_space[key] = spaces.Box(
                    low=0, high=255, shape=(1,) + new_space[key].shape, dtype=np.uint8
                )
            elif key == 'obj_seg':
                new_space[key] = spaces.Box(
                    low=0, high=1, shape=(1,) + new_space[key].shape, dtype=np.bool_
                )
            else:
                new_space[key] = spaces.Box(
                    low=-np.inf, high=np.inf, shape=(1,) + new_space[key].shape, dtype=np.float32
                )

    def observation(self, observation: dict):
        for key, value in observation.items():
            observation[key] = value[None]
        return observation


class DictObservationStack(gym.ObservationWrapper):
    """Observation wrapper that stacks the observations in a rolling manner.

    For example, if the number of stacks is 4, then the returned observation contains
    the most recent 4 observations. For environment 'Pendulum-v1', the original observation
    is an array with shape [3], so if we stack 4 observations, the processed observation
    has shape [4, 3].

    Note:
        - To be memory efficient, the stacked observations are wrapped by :class:`LazyFrame`.
        - The observation space must be :class:`Box` type. If one uses :class:`Dict`
          as observation space, it should apply :class:`FlattenObservation` wrapper first.
          - After :meth:`reset` is called, the frame buffer will be filled with the initial observation. I.e. the observation returned by :meth:`reset` will consist of ``num_stack`-many identical frames,

    """

    def __init__(
        self,
        env: gym.Env,
        num_stack: int,
        lz4_compress: bool = False,
    ):
        """Observation wrapper that stacks the observations in a rolling manner.

        Args:
            env (Env): The environment to apply the wrapper
            num_stack (int): The number of frames to stack
            lz4_compress (bool): Use lz4 to compress the frames internally
        """
        super().__init__(env)
        self.num_stack = num_stack
        self.lz4_compress = lz4_compress

        self.frames = OrderedDict()

        for key in self.observation_space.keys():
            low = np.repeat(self.observation_space[key].low[np.newaxis, ...], num_stack, axis=0)
            high = np.repeat(
                self.observation_space[key].high[np.newaxis, ...], num_stack, axis=0
            )

            self.observation_space[key] = spaces.Box(
                low=low, high=high, dtype=self.observation_space[key].dtype
            )
            self.frames[key] = deque(maxlen=num_stack)

    def observation(self, observation: dict):
        """Converts the wrappers current frames to lazy frames.

        Args:
            observation: Ignored

        Returns:
            :class:`LazyFrames` object for the wrapper's frame buffer,  :attr:`self.frames`
        """
        # assert len(self.frames) == self.num_stack, (len(self.frames), self.num_stack)
        for key in observation.keys():
            observation[key] = LazyFrames(list(self.frames[key]), self.lz4_compress)
        return observation

    def step(self, action):
        """Steps through the environment, appending the observation to the frame buffer.

        Args:
            action: The action to step through the environment with

        Returns:
            Stacked observations, reward, terminated, truncated, and information from the environment
        """
        observation, reward, done, info = self.env.step(action)
        for key in observation.keys():
            self.frames[key].append(observation[key])
        return self.observation(observation), reward, done, info

    def reset(self, **kwargs):
        """Reset the environment with kwargs.

        Args:
            **kwargs: The kwargs for the environment reset

        Returns:
            The stacked observations
        """
        obs = self.env.reset(**kwargs)

        for key in obs.keys():
            [self.frames[key].append(obs[key]) for _ in range(self.num_stack)]

        return self.observation(obs)
