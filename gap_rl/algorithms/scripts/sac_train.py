from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union
import os
import yaml
import time
import argparse
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

from gap_rl import ALGORITHM_DIR, RLMODEL_DIR
from gap_rl.envs import *
from gap_rl.utils.common import setup_seed
from gap_rl.utils.wrappers.common import NormalizeBoxActionWrapper
from gap_rl.utils.wrappers.record import RecordEpisode
from gap_rl.algorithms.rl_utils import sb3_make_env, sb3_make_multienv
from gap_rl.algorithms.rl_utils import (
    CustomObjPNExtractor,
    CustomGraspExtractor,
    CustomGraspPointExtractor,
    CustomGraspPointGroupExtractor,
)
from custom_sac import CustomSAC

from stable_baselines3 import SAC
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor


extractor_aliases: Dict[str, Type[BaseFeaturesExtractor]] = {
    "state_grasp9d": CustomGraspExtractor,
    # CustomGraspPointExtractor, CustomGraspPointGroupExtractor
    "state_egopoints": CustomGraspPointGroupExtractor,
    "state_eogkeypoints": CustomGraspPointGroupExtractor,
    "state_objpoints_rt": CustomObjPNExtractor,
}


@hydra.main(version_base=None, config_path="../alg_config", config_name="indy7_sac")
def main(cfg: DictConfig):
    np.set_printoptions(suppress=True, precision=4)
    # parser = argparse.ArgumentParser(description="train SAC")

    # parser.add_argument("--config-name", type=str, default="default", help="train config file.")
    # parser.add_argument("--exp-suffix", type=str, default=None, help="exp detail indications.")
    # parser.add_argument("--timestamp", type=str, default=None, help="exp time stamp.")
    # parser.add_argument("--distill", action='store_true')

    # parser.add_argument("--group_name", type=str, default="test", help="wandb group name.")
    # parser.add_argument("--exp_name", type=str, default="test", help="wandb experiment name.")


    # args = parser.parse_args()

    # config_file = ALGORITHM_DIR / f"config/{args.config_name}.yaml"
    # with open(config_file, "r", encoding="utf-8") as fin:
    #     cfg = yaml.load(fin, Loader=yaml.FullLoader)

    robot_cfg = cfg.robot
    alg_cfg = cfg.alg 

    is_goal_aux = cfg.get("goal_aux", False)
    share_feat = cfg.get("share_feat", True)


    # # FIXME: 디버깅
    # cfg['train_procs'] = 1


    env_cfg_file = ALGORITHM_DIR / f"config/env_settings.yaml"
    with open(env_cfg_file, "r", encoding="utf-8") as fin:
        env_cfg = yaml.load(fin, Loader=yaml.FullLoader)
    env_id = env_cfg["ycb_train"]["env_id"]
    model_ids = env_cfg["ycb_train"]["model_ids"]

    # seed = np.random.RandomState().randint(2**32)
    seed = cfg.get("seed", 1)
    print("experiment random seed: ", seed)
    setup_seed(seed)

    rl_feat_extract_class = extractor_aliases[robot_cfg.obs_mode]

    # Create log dir
    vary_str = "vary" if robot_cfg["vary_speed"] else "fix"
    exp_suffix = f"YCB{len(model_ids)}_{robot_cfg['num_grasps']}_{robot_cfg['gen_traj_mode']}_{vary_str}"
    time_stamp =  time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_dir = f"{time_stamp}_sac{robot_cfg['train_procs']}_{robot_cfg['obs_mode']}_{robot_cfg['control_mode']}_{exp_suffix}"
    robot_cfg["log_dir"] = log_dir
    os.makedirs(log_dir, exist_ok=True)
    # with open(log_dir + "/config.yaml", "w", encoding="utf-8") as file:
    #     yaml.dump(cfg, file)
    with open(os.path.join(log_dir, "config.yaml"), "w") as file:
        file.write(OmegaConf.to_yaml(cfg))




    # from stable_baselines3.common.vec_env import DummyVecEnv
    # env = sb3_make_multienv(
    #     env_id=env_id,
    #     robot_id=cfg["robot_id"],
    #     robot_init_qpos_noise=cfg["robot_init_qpos_noise"],
    #     shader_dir=cfg["shader_dir"],
    #     model_ids=model_ids,
    #     num_grasps=cfg["num_grasps"],
    #     num_grasp_points=cfg["num_grasp_points"],
    #     grasp_points_mode=cfg["grasp_points_mode"],
    #     obj_init_rot_z=cfg["obj_init_rot_z"],
    #     obj_init_rot=cfg["obj_init_rot"],
    #     goal_thresh=cfg["goal_thresh"],
    #     robot_x_offset=cfg["robot_x_offset"],
    #     gen_traj_mode=cfg["gen_traj_mode"],
    #     vary_speed=cfg["vary_speed"],
    #     grasp_select_mode=cfg["grasp_select_mode"],
    #     obs_mode=cfg["obs_mode"],
    #     control_mode=cfg["control_mode"],
    #     reward_mode=cfg["reward_mode"],
    #     sim_freq=cfg["sim_freq"],
    #     control_freq=cfg["control_freq"],
    #     device=cfg["device"],
    #     rank=0,
    #     seed=seed,
    # )

    # vec_env = DummyVecEnv([env])


    vec_env = SubprocVecEnv(
        [
            sb3_make_multienv(
                env_id=env_id,
                robot_id=robot_cfg["robot_id"],
                robot_init_qpos_noise=robot_cfg["robot_init_qpos_noise"],
                shader_dir=robot_cfg["shader_dir"],
                model_ids=model_ids,
                num_grasps=robot_cfg["num_grasps"],
                num_grasp_points=robot_cfg["num_grasp_points"],
                grasp_points_mode=robot_cfg["grasp_points_mode"],
                obj_init_rot_z=robot_cfg["obj_init_rot_z"],
                obj_init_rot=robot_cfg["obj_init_rot"],
                goal_thresh=robot_cfg["goal_thresh"],
                robot_x_offset=robot_cfg["robot_x_offset"],
                gen_traj_mode=robot_cfg["gen_traj_mode"],
                vary_speed=robot_cfg["vary_speed"],
                grasp_select_mode=robot_cfg["grasp_select_mode"],
                obs_mode=robot_cfg["obs_mode"],
                control_mode=robot_cfg["control_mode"],
                reward_mode=robot_cfg["reward_mode"],
                sim_freq=robot_cfg["sim_freq"],
                control_freq=robot_cfg["control_freq"],
                device=robot_cfg["device"],
                rank=i,
                seed=seed,
            )
            for i in range(robot_cfg["train_procs"])
        ],
        start_method="spawn",
    )
    vec_env = VecMonitor(vec_env, log_dir)

    for obs_key, box_space in vec_env.observation_space.items():
        print(f"{obs_key}: {box_space.shape} ")
    print("Action Space: ", vec_env.action_space)

    # setup callbacks
    # checkpoint_callback = CheckpointCallback(save_freq=400000 // cfg.get("train_procs", 1), save_path=log_dir)
    # FIXME: 디버깅중
    checkpoint_callback = CheckpointCallback(save_freq=2500, save_path=log_dir)
    # callback = CallbackList([checkpoint_callback])
 
    # 1) wandb init

    plain_cfg = OmegaConf.to_container(robot_cfg, resolve=True)
    wandb.init(
        project="gap-rl",
        name=f"{time_stamp}_sac_{robot_cfg['obs_mode']}_{exp_suffix}",
        config=plain_cfg,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    )
    wandb_callback = WandbCallback(
        # 모델 체크포인트도 wandb에 업로드
        model_save_path=os.path.join(log_dir, "wandb_models"),
        verbose=2,
    )
    callback = CallbackList([checkpoint_callback, wandb_callback])

    # set up logger
    new_logger = configure(log_dir, ["stdout", "csv", "log", "tensorboard"])

    # check_env(env)
    if is_goal_aux:
        model = CustomSAC(
            "CustomSACPolicy",
            vec_env,
            batch_size=alg_cfg.batch_size,  # 1024, 400
            ent_coef="auto_0.2",
            target_entropy=alg_cfg.target_entropy,
            learning_rate=alg_cfg.learning_rate,
            gamma=alg_cfg.gamma,
            train_freq=16,  # 4, 64
            gradient_steps=16,  # 2, 4
            buffer_size=100000,
            learning_starts=800,
            use_sde=True,
            policy_kwargs=dict(
                # optimizer_class=torch.optim.AdamW,
                log_std_init=-3.67,
                net_arch=[256, 256],
                features_extractor_class=rl_feat_extract_class,
                features_extractor_kwargs=None,
                normalize_images=False,
                share_features_extractor=share_feat,
                extra_pred_dim=9,
            ),
            tensorboard_log=log_dir + "sac_opendoor_tb/",
            seed=seed,
            device=cfg["device"],
            # distill=args.distill,
            verbose=1,
        )
    else:
        model = SAC(
            "MultiInputPolicy",
            vec_env,
            batch_size=alg_cfg.batch_size,  # 1024, 400
            ent_coef="auto_0.2",
            target_entropy=alg_cfg.target_entropy,
            learning_rate=alg_cfg.learning_rate,
            gamma=alg_cfg.gamma,
            train_freq=16,  # 4, 64
            gradient_steps=16,  # 2, 4
            buffer_size=100000,
            learning_starts=800,
            use_sde=True,
            policy_kwargs=dict(
                # optimizer_class=torch.optim.AdamW,
                log_std_init=-3.67,
                net_arch=[256, 256],
                features_extractor_class=rl_feat_extract_class,
                features_extractor_kwargs=None,
                normalize_images=False,
                share_features_extractor=share_feat,
            ),
            tensorboard_log=log_dir + "sac_opendoor_tb/",
            seed=seed,
            device=cfg["device"],
            verbose=1,
        )
    # Set new logger
    model.set_logger(new_logger)
    model.learn(
        total_timesteps=alg_cfg.total_timesteps,
        # callback=[checkpoint_callback],
        callback=callback,
    )
    # model.save_replay_buffer(log_dir + "/sac_replay_buffer")

    # set evaluation
    eval_seed = 1029
    setup_seed(eval_seed)
    evalenv = sb3_make_env(
        env_id=env_id,
        robot_id=robot_cfg["robot_id"],
        robot_init_qpos_noise=robot_cfg["robot_init_qpos_noise"],
        shader_dir=robot_cfg["shader_dir"],
        model_ids=model_ids,
        num_grasps=robot_cfg["num_grasps"],
        num_grasp_points=robot_cfg["num_grasp_points"],
        grasp_points_mode=robot_cfg["grasp_points_mode"],
        obj_init_rot_z=robot_cfg["obj_init_rot_z"],
        obj_init_rot=robot_cfg["obj_init_rot"],
        goal_thresh=robot_cfg["goal_thresh"],
        robot_x_offset=robot_cfg["robot_x_offset"],
        gen_traj_mode=robot_cfg["gen_traj_mode"],
        vary_speed=robot_cfg["vary_speed"],
        grasp_select_mode=robot_cfg["grasp_select_mode"],
        obs_mode=robot_cfg["obs_mode"],
        control_mode=robot_cfg["control_mode"],
        reward_mode=robot_cfg["reward_mode"],
        sim_freq=robot_cfg["sim_freq"],
        control_freq=robot_cfg["control_freq"],
        device=robot_cfg["device"],
        seed=eval_seed,
    )
    print("set eval env with seed: ", eval_seed)

    success_rates = defaultdict(list)
    success_steps = defaultdict(list)
    for eval_step in robot_cfg["eval_steps"]:
        model_path = f"{log_dir}/rl_model_{eval_step}_steps"
        rl_model = SAC.load(model_path, env=evalenv, print_system_info=True)
        for model_id in model_ids:
            result_dir = f"{log_dir}/result/{model_id}/{eval_step / 1000000}M/"
            os.makedirs(result_dir, exist_ok=True)
            record_env = RecordEpisode(
                env=evalenv,
                output_dir=result_dir,
                save_trajectory=False,
                trajectory_name=None,
                save_video=True,
                info_on_video=True,
                render_mode="agent_cameras",
                save_on_reset=True,
                clean_on_close=True,
            )
            record_env.seed(eval_seed)

            is_success = []
            exp_steps = []
            with torch.no_grad():
                for num in tqdm(
                    range(robot_cfg["test_num"]), desc=f"Processing {eval_step / 1000000}M", colour="green", leave=True
                ):
                    
                    obs = record_env.reset(model_id=model_id)
                    epi_seed = record_env.unwrapped._episode_seed
                    success_flag = False
                    total_steps = 0
                    for step in tqdm(range(robot_cfg["max_steps"]), colour="red", leave=False):
                        action, _states = rl_model.predict(obs, deterministic=True)
                        obs, rewards, done, info = record_env.step(action)
                        if not success_flag and info["is_success"]:
                            success_flag = True
                            total_steps = step
                    is_success.append(np.array(success_flag).astype(float))
                    exp_steps.append(total_steps)
                    suffix = f"success_{total_steps}" if success_flag else "failure"
                    record_env.set_video_suffix(f"{epi_seed}_" + suffix)
            success_rate = np.sum(is_success) / robot_cfg["test_num"]
            exp_steps = np.array(exp_steps)
            success_mask = exp_steps > 0
            if np.any(success_mask):
                mean_steps = np.mean(exp_steps[success_mask])
            else:
                mean_steps = robot_cfg["max_steps"]
            print(f"{eval_step}, {model_id}, success_rate, mean_steps: {success_rate}, {mean_steps}")
            success_rates[model_id].append(success_rate)
            success_steps[model_id].append(mean_steps)

    plots_perfig = 4
    ids = np.arange(len(model_ids))
    id_seg = ids[::plots_perfig][1:]
    ids_seg = np.split(ids, id_seg)
    for k, id_range in enumerate(ids_seg):
        fig, ax = plt.subplots()
        for model_id in np.array(model_ids)[id_range]:
            ax.plot(np.array(robot_cfg["eval_steps"]) / 1000000, success_rates[model_id], "-", label=model_id)
        ax.legend(loc="lower left")
        ax.set_xlabel("eval_model_steps(M)")
        ax.set_ylabel("success rate")
        ax2 = ax.twinx()
        for model_id in np.array(model_ids)[id_range]:
            ax2.plot(np.array(robot_cfg["eval_steps"]) / 1000000, success_steps[model_id], "--", label=model_id)
        ax2.legend(loc="upper right")
        ax2.set_ylabel("mean steps")
        plt.savefig(f"{log_dir}/result/sr-evalresults{k}.png")

    print(model_ids)
    print(robot_cfg["eval_steps"])
    print(success_rates)
    print(success_steps)
    sr_2m = np.array([success_rates[model_id][-1] for model_id in model_ids])
    st_2m = np.array([success_steps[model_id][-1] for model_id in model_ids])
    mean_sr, mean_st = np.mean(sr_2m).round(3), np.mean(st_2m).round(1)
    std_sr, std_st = np.std(sr_2m).round(3), np.std(st_2m).round(1)
    with open(f"{log_dir}/result/success_rates_steps.txt", "w") as f:
        f.write(f"eval steps: {robot_cfg['eval_steps']}\n")
        for model_id in model_ids:
            f.write(f"{model_id}\n")
            f.write(f"success rates: {success_rates[model_id]}")
            f.write(f"success steps: {success_steps[model_id]}\n")
        f.write(f"Total success rates (mean, std): ({mean_sr}, {std_sr})\n")
        f.write(f"Total success steps (mean, std): ({mean_st}, {std_st})\n")


if __name__ == "__main__":
    main()