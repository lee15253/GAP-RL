import copy
import time
from pathlib import Path

import gym
import h5py
import numpy as np
from gap_rl import get_commit_info, logger
from gym import spaces

from ..common import extract_scalars_from_info, flatten_dict_keys
from ..io_utils import dump_json
from ..visualization.misc import images_to_video, put_info_on_image


def parse_env_info(env: gym.Env):
    # spec can be None if not initialized from gym.make
    env = env.unwrapped
    if env.spec is None:
        return None
    if hasattr(env.spec, "_kwargs"):
        # gym<=0.21
        env_kwargs = env.spec._kwargs
    else:
        # gym>=0.22
        env_kwargs = env.spec.kwargs
    return dict(
        env_id=env.spec.id,
        max_episode_steps=env.spec.max_episode_steps,
        env_kwargs=env_kwargs,
    )


def clean_trajectories(h5_file: h5py.File, json_dict: dict, prune_empty_action=True):
    """Clean trajectories by renaming and pruning trajectories in place.

    After cleanup, trajectory names are consecutive integers (traj_0, traj_1, ...),
    and trajectories with empty action are pruned.

    Args:
        h5_file: raw h5 file
        json_dict: raw JSON dict
        prune_empty_action: whether to prune trajectories with empty action
    """
    json_episodes = json_dict["episodes"]
    assert len(h5_file) == len(json_episodes)

    # Assumes each trajectory is named "traj_{i}"
    prefix_length = len("traj_")
    ep_ids = sorted([int(x[prefix_length:]) for x in h5_file.keys()])

    new_json_episodes = []
    new_ep_id = 0

    for i, ep_id in enumerate(ep_ids):
        traj_id = f"traj_{ep_id}"
        ep = json_episodes[i]
        assert ep["episode_id"] == ep_id
        new_traj_id = f"traj_{new_ep_id}"

        if prune_empty_action and ep["elapsed_steps"] == 0:
            del h5_file[traj_id]
            continue

        if new_traj_id != traj_id:
            ep["episode_id"] = new_ep_id
            h5_file[new_traj_id] = h5_file[traj_id]
            del h5_file[traj_id]

        new_json_episodes.append(ep)
        new_ep_id += 1

    json_dict["episodes"] = new_json_episodes


class RecordEpisode(gym.Wrapper):
    """Record trajectories or videos for episodes.
    The trajectories are stored in HDF5.

    Args:
        env: gym.Env
        output_dir: output directory
        save_trajectory: whether to save trajectory
        trajectory_name: name of trajectory file (.h5). Use timestamp if not provided.
        save_video: whether to save video
        render_mode: rendering mode passed to `env.render`
        save_on_reset: whether to save the previous trajectory automatically when resetting.
            If True, the trajectory with empty transition will be ignored automatically.
        clean_on_close: whether to rename and prune trajectories when closed.
            See `clean_trajectories` for details.
    """

    def __init__(
        self,
        env,
        output_dir,
        save_trajectory=True,
        trajectory_name=None,
        save_video=True,
        info_on_video=False,
        render_mode="rgb_array",
        save_on_reset=True,
        clean_on_close=True,
    ):
        super().__init__(env)

        self.output_dir = Path(output_dir)
        if save_trajectory or save_video:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_on_reset = save_on_reset

        self.suffix = ""
        self._elapsed_steps = 0
        self._episode_id = -1
        self._episode_data = []
        self._episode_info = {}

        self.save_trajectory = save_trajectory
        self.clean_on_close = clean_on_close
        if self.save_trajectory:
            if not trajectory_name:
                trajectory_name = time.strftime("%Y%m%d_%H%M%S")

            self._h5_file = h5py.File(self.output_dir / f"{trajectory_name}.h5", "w")

            # Use a separate json to store non-array data
            self._json_path = self._h5_file.filename.replace(".h5", ".json")
            self._json_data = dict(
                env_info=parse_env_info(self.env),
                commit_info=get_commit_info(),
                episodes=[],
            )

        self.save_video = save_video
        self.info_on_video = info_on_video
        self.bk_render_mode = render_mode
        self._render_images = []

        self.init_state_only = False

    def reset_output_dir(self, output_dir):
        self._episode_id = -1
        cur_output_dir = Path(output_dir)
        if self.output_dir != output_dir:
            self.output_dir = cur_output_dir

    def reset(self, **kwargs):
        if self.save_on_reset and self._episode_id >= 0:
            if self._elapsed_steps == 0:
                self._episode_id -= 1
            self.flush_trajectory(ignore_empty_transition=True)
            self.flush_video(ignore_empty_transition=True)

        # Clear cache
        self._elapsed_steps = 0
        self._episode_id += 1
        self._episode_data = []
        self._episode_info = {}
        self._render_images = []

        reset_kwargs = copy.deepcopy(kwargs)
        obs = super().reset(**kwargs)

        if self.save_trajectory:
            state = self.env.get_state()
            data = dict(s=state, o=obs, a=None, r=None, done=None, info=None)
            self._episode_data.append(data)
            self._episode_info.update(
                episode_id=self._episode_id,
                episode_seed=getattr(self.unwrapped, "_episode_seed", None),
                reset_kwargs=reset_kwargs,
                control_mode=getattr(self.unwrapped, "control_mode", None),
                elapsed_steps=0,
            )

        if self.save_video:
            self._render_images.append(self.env.render(self.bk_render_mode))

        return obs

    def step(self, action):
        # BK fix
        # obs, rew, done, info = super().step(action)
        obs, rew, terminated, truncated, info = super().step(action)
        done = terminated or truncated
        
        self._elapsed_steps += 1

        if self.save_trajectory:
            state = self.env.get_state()
            data = dict(s=state, o=obs, a=action, r=rew, done=done, info=info)
            self._episode_data.append(data)
            self._episode_info["elapsed_steps"] += 1
            self._episode_info["info"] = info

        if self.save_video:
            image = self.env.render(self.bk_render_mode)

            if self.info_on_video:
                # scalar_info = extract_scalars_from_info(info)
                extra_texts = [
                    f"reward: {rew:.3f}",
                    "action: {}".format(",".join([f"{x:.2f}" for x in action])),
                ]
                image = put_info_on_image(image, info, extras=extra_texts)

            self._render_images.append(image)

        return obs, rew, done, info

    def flush_trajectory(self, verbose=False, ignore_empty_transition=False):
        if not self.save_trajectory or len(self._episode_data) == 0:
            return
        if ignore_empty_transition and len(self._episode_data) == 1:
            return

        traj_id = "traj_{}".format(self._episode_id)
        group = self._h5_file.create_group(traj_id, track_order=True)

        # Observations need special processing
        obs = [x["o"] for x in self._episode_data]
        if isinstance(obs[0], dict):
            obs = [flatten_dict_keys(x) for x in obs]
            obs = {k: [x[k] for x in obs] for k in obs[0].keys()}
            obs = {k: np.stack(v) for k, v in obs.items()}
            for k, v in obs.items():
                if "rgb" in k and v.ndim == 4:
                    group.create_dataset(
                        "obs/" + k,
                        data=v,
                        dtype=v.dtype,
                        compression="gzip",
                        compression_opts=5,
                    )
                elif "depth" in k and v.ndim in (3, 4):
                    if not np.all(np.logical_and(v >= 0, v < 2 ** 6)):
                        raise RuntimeError(
                            "The depth map({}) is invalid with min({}) and max({}).".format(
                                k, v.min(), v.max()
                            )
                        )
                    v = (v * (2 ** 10)).astype(np.uint16)
                    group.create_dataset(
                        "obs/" + k,
                        data=v,
                        dtype=v.dtype,
                        compression="gzip",
                        compression_opts=5,
                    )
                elif "seg" in k and v.ndim in (3, 4):
                    # assert np.issubdtype(v.dtype, np.integer), v.dtype
                    group.create_dataset(
                        "obs/" + k,
                        data=v,
                        dtype=v.dtype,
                        compression="gzip",
                        compression_opts=5,
                    )
                else:
                    group.create_dataset("obs/" + k, data=v, dtype=v.dtype)
        elif isinstance(obs[0], np.ndarray):
            obs = np.stack(obs)
            group.create_dataset("obs", data=obs, dtype=obs.dtype)
        else:
            raise NotImplementedError(type(obs[0]))

        if len(self._episode_data) == 1:
            action_space = self.env.action_space
            assert isinstance(action_space, spaces.Box), action_space
            actions = np.empty(
                shape=(0,) + action_space.shape, dtype=action_space.dtype,
            )
            dones = np.empty(shape=(0,), dtype=bool)
        else:
            # Record transitions (ignore the first padded values during reset)
            actions = np.stack([x["a"] for x in self._episode_data[1:]])
            dones = np.stack([x["info"]["is_success"] for x in self._episode_data[1:]])

        # Only support array like states now
        env_states = np.stack([x["s"] for x in self._episode_data])

        # Dump
        group.create_dataset("actions", data=actions, dtype=np.float32)
        group.create_dataset("success", data=dones, dtype=bool)
        if self.init_state_only:
            group.create_dataset("env_init_state", data=env_states[0], dtype=np.float32)
        else:
            group.create_dataset("env_states", data=env_states, dtype=np.float32)

        # Handle JSON
        self._json_data["episodes"].append(self._episode_info)
        dump_json(self._json_path, self._json_data, indent=2)

        if verbose:
            print("Record the {}-th episode".format(self._episode_id))

    def set_video_suffix(self, suffix=""):
        self.suffix = suffix

    def flush_video(self, verbose=False, ignore_empty_transition=False):
        if not self.save_video or len(self._render_images) == 0:
            return
        if ignore_empty_transition and len(self._render_images) == 1:
            return

        video_name = "{}".format(self._episode_id) + "_" + self.suffix
        images_to_video(
            self._render_images,
            str(self.output_dir),
            video_name=video_name,
            fps=5,
            verbose=verbose,
        )

    def close(self) -> None:
        if self.save_trajectory:
            # Handle the last episode only when `save_on_reset=True`
            if self.save_on_reset:
                traj_id = "traj_{}".format(self._episode_id)
                if traj_id in self._h5_file:
                    logger.warning(f"{traj_id} exists in h5.")
                else:
                    self.flush_trajectory(ignore_empty_transition=True)
            if self.clean_on_close:
                clean_trajectories(self._h5_file, self._json_data)
            self._h5_file.close()
        if self.save_video:
            if self.save_on_reset:
                self.flush_video(ignore_empty_transition=True)
        return super().close()
