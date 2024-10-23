import os
import yaml
import time
import glob
import argparse
from tqdm import tqdm
import gym
import numpy as np
import torch
import pickle
from collections import OrderedDict

from gap_rl.envs import *
from gap_rl import LOCALGRASP_DIR, ASSET_DIR, ALGORITHM_DIR
from gap_rl.utils.common import setup_seed
from gap_rl.utils.wrappers.common import NormalizeBoxActionWrapper
from gap_rl.utils.wrappers.observation import StackObservationWrapper, DictObservationStack
from gap_rl.utils.wrappers.record import RecordEpisode

from stable_baselines3 import SAC


if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=4)
    device = "cuda:0"

    parser = argparse.ArgumentParser()
    parser.add_argument("--obj-test-num", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--cam-mode", type=str, default="hand_realsense")
    parser.add_argument("--pc-mode", type=str, default="rt")
    parser.add_argument("--gen-traj-modes", type=str, nargs='+', required=True)  # line circular bezier2d random2d
    parser.add_argument("--obs-mode", type=str, default="state_objpoints_rt")
    parser.add_argument("--eval-datasets", type=str, nargs='+', required=True)  # "ycb_train", "ycb_eval", "graspnet_eval", "acronym_eval"
    parser.add_argument("--timestamp", type=str)
    parser.add_argument("--seeds", type=int, nargs='+', required=True)

    parser.add_argument("--save-video", action='store_true')
    parser.add_argument("--save-feat", action='store_true')
    parser.add_argument("--render", action='store_true')
    parser.add_argument("--add-noise", action='store_true')
    args = parser.parse_args()

    print("experiment random seeds: ", args.seeds)
    for rseed in args.seeds:
        cur_seed = int(rseed)
        setup_seed(cur_seed)
        print(args.eval_datasets, args.gen_traj_modes, cur_seed)
        noise_str = "noise" if args.add_noise else ""

        for eval_dataset in args.eval_datasets:
            log_path = glob.glob(f"{args.timestamp}*")[0]
            config_file = ALGORITHM_DIR / f"scripts/{log_path}/config.yaml"
            with open(config_file, 'r', encoding='utf-8') as fin:
                cfg = yaml.load(fin, Loader=yaml.FullLoader)
            env_cfg_file = ALGORITHM_DIR / f"config/env_settings.yaml"
            with open(env_cfg_file, 'r', encoding='utf-8') as fin:
                env_cfg = yaml.load(fin, Loader=yaml.FullLoader)
            env_id = env_cfg[eval_dataset]['env_id']
            model_ids = env_cfg[eval_dataset]['model_ids']
            is_goal_aux = cfg.get('goal_aux', False)

            if args.cam_mode == 'hand_realsense':
                camera_modes = ["hand_realsense"]
                result_path = f'simLoG_result_onlyhandcam_{eval_dataset}_{args.pc_mode}_{cur_seed}_{noise_str}'
            elif args.cam_mode == 'both':
                camera_modes = ["hand_realsense", "base_kinect"]
                result_path = f'simLoG_result_basehandcam_{eval_dataset}_{args.pc_mode}_{cur_seed}_{noise_str}'
            else:
                raise NotImplementedError(args.cam_mode)

            sr_current_dataset = []
            st_current_dataset = []
            for gen_traj_mode in args.gen_traj_modes:
                env = gym.make(
                    env_id,
                    robot=cfg['robot_id'],
                    robot_init_qpos_noise=cfg['robot_init_qpos_noise'],
                    shader_dir=cfg['shader_dir'],
                    model_ids=model_ids,
                    num_grasps=cfg['num_grasps'],
                    obj_init_rot_z=cfg['obj_init_rot_z'],
                    obj_init_rot=cfg['obj_init_rot'],
                    goal_thresh=cfg['goal_thresh'],
                    robot_x_offset=cfg['robot_x_offset'],
                    gen_traj_mode=gen_traj_mode,
                    vary_speed=cfg['vary_speed'],
                    grasp_select_mode=cfg['grasp_select_mode'],
                    obs_mode=args.obs_mode,
                    control_mode=cfg['control_mode'],
                    reward_mode=cfg['reward_mode'],
                    sim_freq=cfg['sim_freq'],
                    control_freq=cfg['control_freq'],
                    device=cfg["device"],
                )
                env = NormalizeBoxActionWrapper(env)
                para = {
                    "camera_modes": camera_modes,
                    "add_noise": args.add_noise
                }
                env.set_rt_paras(**para)
                record_env = RecordEpisode(
                    env=env,
                    output_dir='./',
                    save_trajectory=False,
                    trajectory_name=None,
                    save_video=args.save_video,
                    info_on_video=False,
                    render_mode="agent_cameras",
                    save_on_reset=True,
                    clean_on_close=True,
                )

                # Note: load RL model, it will change record._main_seed
                model_path = f"{log_path}/rl_model_2000000_steps"
                rl_model = SAC.load(model_path,
                                    env=record_env,
                                    device=device,
                                    print_system_info=True
                                    )

                # print("Observation: ", record_env.observation_space.keys())
                # print("Action Space: ", record_env.action_space)
                record_env.seed(cur_seed)
                # print("+++++++++++++ init env main seed: ", record_env.unwrapped._main_seed)

                success_rates = []
                success_steps = []

                if args.save_feat:
                    pn_feat_list = OrderedDict()

                for model_id in model_ids:
                    result_dir = f"{log_path}/{result_path}/{gen_traj_mode}/{model_id}/"
                    os.makedirs(result_dir, exist_ok=True)
                    record_env.reset_output_dir(result_dir)

                    is_success = []
                    exp_steps = []
                    if args.save_feat:
                        pn_feat_list[model_id] = OrderedDict()

                    for num in tqdm(range(args.obj_test_num), desc=f'Processing 2M model', colour='red', leave=True):
                        t = time.time()
                        # print("+++++++++++++ env main seed: ", record_env.unwrapped._main_seed)
                        obs = record_env.reset(model_id=model_id)
                        epi_seed = record_env.unwrapped._episode_seed
                        # print("+++++++++++++ episode seed: ", epi_seed)
                        # print("*************  points add noise: ", record_env.unwrapped.points_add_noise)
                        # record_env.seed(seed)
                        # init_obj_exist = obs['obj_exist']
                        init_obj_exist = np.any(record_env.unwrapped._get_obj_exist_mask())
                        # print("----- reset time: ", time.time() - t)
                        t = time.time()

                        if args.render:
                            viewer = record_env.render()
                            print("Press [e] to start")
                            while True:
                                if viewer.window.key_down("e"):
                                    break
                                record_env.render()
                        else:
                            record_env.update_render()

                        # print("----- init render time: ", time.time() - t)

                        success_flag = False
                        total_steps = 0
                        if args.save_feat:
                            pn_feat_list[model_id][num] = OrderedDict()
                            pn_feat_list[model_id][num] = OrderedDict()
                            pn_feat_list[model_id][num]["actor_pnfeat"] = []
                            pn_feat_list[model_id][num]["critic_pnfeat"] = []

                        with torch.no_grad():
                            for step in tqdm(range(args.max_steps), desc=f'Processing 2M model', colour='green', leave=False):
                                ## grasps_ee as mat form
                                t = time.time()
                                action, _states = rl_model.predict(obs, deterministic=True)
                                if is_goal_aux:
                                    obs_tensor = {k: torch.tensor(o[None], device=device) for k, o in obs.items()}
                                    action_tensor = torch.tensor(action[None], device=device)
                                    _, pred_pose_actor, pred_target_actor = rl_model.actor.action_log_prob(obs_tensor)
                                    _, pred_pose_critic, pred_target_critic = rl_model.critic(obs_tensor, action_tensor)
                                    pred_grasp_actor_critic = torch.cat((pred_pose_actor, pred_pose_critic), dim=0).cpu().numpy()
                                    para = {"pred_grasp_actor_critic": pred_grasp_actor_critic}
                                    env.set_rt_paras(**para)
                                if args.save_feat:
                                    pn_feat_list[model_id][num]["actor_pnfeat"].append(rl_model.actor.features_forward(obs_tensor)[0].cpu().numpy())
                                    pn_feat_list[model_id][num]["critic_pnfeat"].append(rl_model.critic.features_forward(obs_tensor)[0].cpu().numpy())
                                # print(f"----- step {step}: RL prediction time: ", time.time() - t)

                                t = time.time()
                                if not init_obj_exist:
                                    obs, rewards, dones, info = record_env.step(np.zeros(7))
                                else:
                                    obs, rewards, dones, info = record_env.step(action)
                                # print(f"----- step {step}: env step time: ", time.time() - t)
                                t = time.time()

                                # init_obj_exist = obs['obj_exist']
                                init_obj_exist = np.any(record_env.unwrapped._get_obj_exist_mask())

                                if not success_flag and info['is_success']:
                                    success_flag = True
                                    total_steps = step
                                if args.render:
                                    record_env.render()
                                else:
                                    record_env.update_render()
                                # print(f"----- step {step}: env render time: ", time.time() - t)
                                t = time.time()

                            is_success.append(np.array(success_flag).astype(float))
                            exp_steps.append(total_steps)
                            suffix = f"success_{total_steps}" if success_flag else "failure"
                            record_env.set_video_suffix(f"{epi_seed}_" + suffix)

                        if args.render:
                            viewer = record_env.render(mode="human")
                            print("Press [t] to next test")
                            while True:
                                if viewer.window.key_down("t"):
                                    break
                                record_env.render()

                        print(is_success, exp_steps)

                    success_rate = np.sum(is_success) / args.obj_test_num
                    exp_steps = np.array(exp_steps)
                    success_mask = exp_steps > 0
                    if np.any(success_mask):
                        mean_steps = np.mean(exp_steps[success_mask]).round(4)
                    else:
                        mean_steps = args.max_steps
                    print(f"2M, {model_id}, success_rate, mean_steps: {success_rate}, {mean_steps}")
                    success_rates.append(success_rate)
                    success_steps.append(mean_steps)

                if args.save_feat:
                    with open(f"{log_path}/{result_path}/{gen_traj_mode}/objstate_pnfeat.pkl", "wb") as file:
                        pickle.dump(pn_feat_list, file)

                results = [model_ids, success_rates, success_steps]
                mean_sr, mean_st = np.mean(success_rates).round(3), np.mean(success_steps).round(1)
                std_sr, std_st = np.std(success_rates).round(3), np.std(success_steps).round(1)
                print(model_ids)
                print('success rates (mean, std): ', mean_sr, std_sr)
                print('success steps (mean, std): ', mean_st, std_st)
                with open(f"{log_path}/{result_path}/{gen_traj_mode}/success_rates_steps.txt", "w") as f:
                    for lst in results:
                        f.write(f"{lst}\n")
                    f.write(f"success rates (mean, std): ({mean_sr}, {std_sr})\n")
                    f.write(f"success steps (mean, std): ({mean_st}, {std_st})\n")

                sr_current_dataset.append(mean_sr)
                st_current_dataset.append(mean_st)

            mean_sr_d, mean_st_d = np.mean(sr_current_dataset).round(3), np.mean(st_current_dataset).round(1)
            with open(f"{log_path}/{result_path}/mean_success_rates_steps.txt", "w") as f:
                f.write(f"{args.gen_traj_modes}\n")
                f.write(f"success rates: {sr_current_dataset}\n")
                f.write(f"success steps: {st_current_dataset}\n")
                f.write(f"avg (models) success rates, steps: {mean_sr_d}, {mean_st_d}\n")
