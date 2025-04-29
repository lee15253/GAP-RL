import os

import gym
import time
import yaml
import random
import torch
import numpy as np
from collections import OrderedDict
from gap_rl import ALGORITHM_DIR
from gap_rl.envs.base_env import BaseEnv
from gap_rl.envs.pick_single import PickSingleYCBEnv


def setup_seed(seed=1029):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    MAX_STEPS = 100
    test_num = 50
    render = True
    view_workspace, view_traj, view_grasps, view_obj_bbdx = True, True, True, True
    mode = "ycb_train"  # 'ycb_train', 'ycb_eval', 'graspnet_eval', 'acronym_eval'
    np.set_printoptions(suppress=True, precision=4)
    obs_mode = "state_egopoints"  # state_egopoints, state_objpoints_rt, state_grasp9d
    grasp_select_mode = "near4"  # angle_filter, nearest, random, near4, near4_filter
    control_mode = "pd_ee_delta_pose_euler"  # "pd_ee_delta_pose"

    gen_traj_mode = "bezier2d"  # "random2d", "line", "circular", "bezier2d"

    robot_id = "ur5e_robotiq85_old"
    env_cfg_file = ALGORITHM_DIR / f"config/env_settings.yaml"
    with open(env_cfg_file, "r", encoding="utf-8") as fin:
        env_cfg = yaml.load(fin, Loader=yaml.FullLoader)
    env_id = env_cfg[mode]["env_id"]
    model_ids = env_cfg[mode]["model_ids"]
    # model_ids = ['006_mustard_bottle']
    print(env_id)
    print(model_ids)

    seed = np.random.RandomState().randint(2**32)
    print("experiment random seed: ", seed)
    setup_seed(seed)
    np.set_printoptions(suppress=True, precision=3)
    env = gym.make(
        env_id,
        shader_dir="ibl",
        robot=robot_id,
        model_ids=model_ids,
        obj_init_rot_z=True,
        obs_mode=obs_mode,
        reward_mode="dense",
        control_mode=control_mode,
        robot_x_offset=0.56,
        sim_freq=150,
        control_freq=5,
        vary_speed=True,
        num_grasps=40,
        gen_traj_mode=gen_traj_mode,
        grasp_select_mode=grasp_select_mode
    )
    env.seed(seed)
    print(env.action_space)

    keys = ["reset_time", "step_time"]
    time_dict = OrderedDict([kk, np.zeros(test_num)] for kk in keys)
    for num in range(test_num):
        start_time = time.time()
        cur_id = int(num % len(model_ids))
        print("cur model id: ", cur_id, model_ids[cur_id])
        _ = env.reset(model_id=model_ids[cur_id])
        if render:
            viewer = env.render(view_workspace=view_workspace, view_traj=view_traj, view_grasps=view_grasps, view_obj_bbdx=view_obj_bbdx)
            # print("Press [e] to start")
            # while True:
            #     if viewer.window.key_down("e"):
            #         break
            #     env.render(view_workspace=view_workspace, view_traj=view_traj, view_grasps=view_grasps, view_obj_bbdx=view_obj_bbdx)
            env.render(view_workspace=view_workspace, view_traj=view_traj, view_grasps=view_grasps, view_obj_bbdx=view_obj_bbdx)

        time_dict["reset_time"][num] = time.time() - start_time
        cnt = 0
        epi_start_time = time.time()
        for step in range(MAX_STEPS):
            t = time.time()
            obs, rew, done, info = env.step(np.zeros(env.agent.action_space.sample().shape))
            print("++ ", time.time() - t)
            if render:
                # print("Press [c] to continue")
                # while True:
                #     if viewer.window.key_down("c"):
                #         break
                #     env.render(view_workspace=True, view_traj=True, view_grasps=True)
                env.render(view_workspace=view_workspace, view_traj=view_traj, view_grasps=view_grasps, view_obj_bbdx=view_obj_bbdx)
            cnt += 1
            print("test ind, steps: ", num, cnt)
        time_dict["step_time"][num] = (time.time() - epi_start_time) / cnt

    import matplotlib.pyplot as plt
    color_type = ["red", "blue"]
    fig, ax = plt.subplots()
    for key in time_dict.keys():
        ax.plot(np.arange(test_num), time_dict[key], color=color_type.pop(0), label=key)
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
