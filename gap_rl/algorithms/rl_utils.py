import gym
import torch

from gap_rl.algorithms.Networks.pointnet import PointNetfeat, GraspPointAppGroup
from gap_rl.utils.wrappers.common import NormalizeBoxActionWrapper
from gap_rl.utils.wrappers.observation import StackObservationWrapper, DictObservationStack

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


def sb3_make_env(
        env_id,
        robot_id,
        robot_init_qpos_noise,
        shader_dir,
        model_ids,
        num_grasps,
        num_grasp_points,
        grasp_points_mode,
        obj_init_rot_z,
        obj_init_rot,
        goal_thresh,
        robot_x_offset,
        gen_traj_mode,
        vary_speed,
        grasp_select_mode,
        obs_mode,
        control_mode,
        reward_mode,
        sim_freq,
        control_freq,
        device="cpu",
        n_stack=0,
        norm_action=True,
        seed=0,
):
    env = gym.make(
        env_id,
        robot=robot_id,
        robot_init_qpos_noise=robot_init_qpos_noise,
        shader_dir=shader_dir,
        num_grasps=num_grasps,
        num_grasp_points=num_grasp_points,
        grasp_points_mode=grasp_points_mode,
        model_ids=model_ids,
        obj_init_rot_z=obj_init_rot_z,
        obj_init_rot=obj_init_rot,
        goal_thresh=goal_thresh,
        robot_x_offset=robot_x_offset,
        gen_traj_mode=gen_traj_mode,
        vary_speed=vary_speed,
        grasp_select_mode=grasp_select_mode,
        obs_mode=obs_mode,
        control_mode=control_mode,
        reward_mode=reward_mode,
        sim_freq=sim_freq,
        control_freq=control_freq,
        device=device,
    )
    if n_stack:
        env = DictObservationStack(env, num_stack=n_stack)
    if norm_action:
        env = NormalizeBoxActionWrapper(env)
    env.seed(seed)
    return env


def sb3_make_multienv(
        env_id,
        robot_id,
        robot_init_qpos_noise,
        shader_dir,
        model_ids,
        num_grasps,
        num_grasp_points,
        grasp_points_mode,
        obj_init_rot_z,
        obj_init_rot,
        goal_thresh,
        robot_x_offset,
        gen_traj_mode,
        vary_speed,
        grasp_select_mode,
        obs_mode,
        control_mode,
        reward_mode,
        sim_freq=300,
        control_freq=20,
        device="cpu",
        stack_obs=False,
        norm_action=True,
        rank=0,
        seed=0
):
    def _init():
        env = gym.make(
            env_id,
            robot=robot_id,
            robot_init_qpos_noise=robot_init_qpos_noise,
            shader_dir=shader_dir,
            num_grasps=num_grasps,
            num_grasp_points=num_grasp_points,
            grasp_points_mode=grasp_points_mode,
            model_ids=model_ids,
            obj_init_rot_z=obj_init_rot_z,
            obj_init_rot=obj_init_rot,
            goal_thresh=goal_thresh,
            robot_x_offset=robot_x_offset,
            gen_traj_mode=gen_traj_mode,
            vary_speed=vary_speed,
            grasp_select_mode=grasp_select_mode,
            obs_mode=obs_mode,
            control_mode=control_mode,
            reward_mode=reward_mode,
            sim_freq=sim_freq,
            control_freq=control_freq,
            device=device,
            renderer_kwargs={"offscreen_only":True, 'device':'cuda0'}  # FIXME: Kaist server            
        )
        # Important: use a different seed for each environment
        if stack_obs:
            env = StackObservationWrapper(env)
        if norm_action:
            env = NormalizeBoxActionWrapper(env)
        env.seed(seed + rank)
        return env

    return _init


class CustomGraspPointExtractor(BaseFeaturesExtractor):
    def __init__(
            self,
            observation_space: gym.spaces.Dict,
            device="cuda:0",
    ):
        super(CustomGraspPointExtractor, self).__init__(observation_space, features_dim=1)

        self.pn = PointNetfeat(
            in_ch=3,
            global_feat=True,
            mlp_specs=[32, 64, 256],
            xyz_transform=False,
            feature_transform=False,
        ).to(torch.device(device))
        self.pn.train()
        state_input_dim = 20
        self.state_map = torch.nn.Linear(state_input_dim, 128, bias=True).to(torch.device(device))
        torch.nn.init.xavier_uniform_(self.state_map.weight, gain=1)
        torch.nn.init.constant_(self.state_map.bias, 0)
        self.state_map.train()

        # Update the features dim manually
        self._features_dim = 256 + 128

    def forward(self, observations) -> torch.Tensor:
        gripper_pos = observations["gripper_pos"]  # (N, 2)
        ee_pose_base = observations["tcp_pose"]  # (N, 6)
        action = observations["action"]  # (N, 7)
        grasp_exist = observations["grasp_exist"]  # (N, 5)
        gripper_pts_diff = observations["gripper_pts_diff"]  # (N, ng, k, 3)
        bs, ng, k, _ = gripper_pts_diff.shape
        gripper_pts_diff = gripper_pts_diff.reshape(bs, ng * k, -1)

        ## PointNet to get 256 feat
        pn_feature, _, _ = self.pn(
            gripper_pts_diff.transpose(2, 1).contiguous()
        )  # (N, 256)
        ## other state
        state = torch.cat([ee_pose_base, gripper_pos, action, grasp_exist], dim=1)
        state_feature = self.state_map(state)

        return torch.cat((state_feature, pn_feature), dim=1)


class CustomGraspPointGroupExtractor(BaseFeaturesExtractor):
    def __init__(
            self,
            observation_space: gym.spaces.Dict,
            group_center_cat: bool = False,
            device="cuda:0",
    ):
        super(CustomGraspPointGroupExtractor, self).__init__(observation_space, features_dim=1)

        self.group_center_cat = group_center_cat
        self.gpcg = GraspPointAppGroup(
            in_ch=3,
            graspgroup_mlp_specs=[16, 32],
            group_mlp_specs=[64, 256],
        ).to(torch.device(device))
        self.gpcg.train()
        state_input_dim = 23 if group_center_cat else 20
        self.state_map = torch.nn.Linear(state_input_dim, 128, bias=True).to(torch.device(device))
        torch.nn.init.xavier_uniform_(self.state_map.weight, gain=1)
        torch.nn.init.constant_(self.state_map.bias, 0)
        self.state_map.train()

        # Update the features dim manually
        self._features_dim = 256 + 128

    def forward(self, observations) -> torch.Tensor:
        gripper_pos = observations["gripper_pos"]  # (N, 2)
        ee_pose_base = observations["tcp_pose"]  # (N, 6)
        action = observations["action"]  # (N, 7)
        grasp_exist = observations["grasp_exist"]  # (N, 5)
        
        # 20개의 sampled gaussian
        origin_gripper_pts = observations["origin_gripper_pts"]  # (N, k, 3)
        # (40,3)의 grasp후보군이 가우시안만큼 이동한 것
        gripper_pts_diff = observations["gripper_pts_diff"]  # (N, ng, k, 3)
        bs, ng, k, _ = gripper_pts_diff.shape

        ## other state

        # RL input: O_grasp + O_state
        # O_grasp
        pn_feature = self.gpcg(origin_gripper_pts, gripper_pts_diff)  # (N, 256)
        state = torch.cat([ee_pose_base, gripper_pos, action, grasp_exist], dim=1)
        # O_state
        state_feature = self.state_map(state)

        return torch.cat((state_feature, pn_feature), dim=1)


class CustomStateObjPNExtractor(BaseFeaturesExtractor):
    def __init__(
            self,
            observation_space: gym.spaces.Dict,
            device="cuda:0",
    ):
        super(CustomStateObjPNExtractor, self).__init__(observation_space, features_dim=1)

        self.pn = PointNetfeat(
            in_ch=3 + 2,
            global_feat=True,
            mlp_specs=[32, 64, 256],
            xyz_transform=False,
            feature_transform=False,
        ).to(torch.device(device))
        self.pn.train()
        self.state_map = torch.nn.Linear(15, 128, bias=True).to(torch.device(device))
        torch.nn.init.xavier_uniform_(self.state_map.weight, gain=1)
        torch.nn.init.constant_(self.state_map.bias, 0)
        self.state_map.train()

        # Update the features dim manually
        self._features_dim = 256 + 128

    def forward(self, observations) -> torch.Tensor:
        gripper_pos = observations["gripper_pos"]  # (N, 2)
        ee_pose_base = observations["tcp_pose"]  # (N, 6)
        action = observations["action"]  # (N, 7)
        obj_pc_ee = observations["obj_pc_ee"]  # (N, 256, 3)
        gripper_pad_pts_ee = observations["gripper_pad_pts_ee"]  # (N, 40, 3)
        bs = gripper_pos.shape[0]
        obj_pad_pc_ee = torch.cat((obj_pc_ee, gripper_pad_pts_ee), dim=1)
        cls_labels = torch.cat((
            torch.tensor([[[1., 0.]]]).repeat(bs, obj_pc_ee.shape[1], 1),
            torch.tensor([[[0., 1.]]]).repeat(bs, gripper_pad_pts_ee.shape[1], 1)
        ), dim=1).to(obj_pc_ee.device)
        obj_pad_pc_cls = torch.cat((obj_pad_pc_ee, cls_labels), dim=2)   # (N, 256+40, 3+2)

        ## PointNet to get 256 feat
        pn_feature, _, _ = self.pn(
            obj_pad_pc_cls.transpose(2, 1).contiguous()
        )  # (N, 256)
        ## other state
        state = torch.cat([ee_pose_base, gripper_pos, action], dim=1)
        state_feature = self.state_map(state)

        return torch.cat((state_feature, pn_feature), dim=1)


class CustomObjPNExtractor(BaseFeaturesExtractor):
    def __init__(
            self,
            observation_space: gym.spaces.Dict,
            device="cuda:0",
    ):
        super(CustomObjPNExtractor, self).__init__(observation_space, features_dim=1)

        self.pn = PointNetfeat(
            in_ch=3+15,
            global_feat=True,
            mlp_specs=[32, 64, 256],
            xyz_transform=False,
            feature_transform=False,
        ).to(torch.device(device))
        self.pn.train()

        # Update the features dim manually
        self._features_dim = 256

    def forward(self, observations) -> torch.Tensor:
        action = observations["action"]  # (N, 7)
        tcp_pose = observations["tcp_pose"]  # (N, 6)
        gripper_pos = observations["gripper_pos"]  # (N, 2)
        obj_pc_ee = observations["obj_pc_ee"]  # (N, 256, 3)
        bs, num, _ = obj_pc_ee.shape

        ## PointNet to get 256 feat
        state = torch.cat((tcp_pose, gripper_pos, action), dim=1)  # (N, 15)
        pn_state = torch.cat((obj_pc_ee, state.unsqueeze(1).repeat(1, num, 1)), dim=2)  # (N, 256, 3+15)
        pn_feature, _, _ = self.pn(
            pn_state.transpose(2, 1).contiguous()
        )  # (N, 256)

        return pn_feature


class CustomGraspExtractor(BaseFeaturesExtractor):
    def __init__(
            self,
            observation_space: gym.spaces.Dict,
            device="cuda:0",
    ):
        super(CustomGraspExtractor, self).__init__(observation_space, features_dim=1)

        self.pn = PointNetfeat(
            in_ch=9,
            global_feat=True,
            mlp_specs=[32, 64, 256],
            xyz_transform=False,
            feature_transform=False,
        ).to(torch.device(device))
        self.pn.train()
        self.state_map = torch.nn.Linear(20, 128, bias=True).to(torch.device(device))
        torch.nn.init.xavier_uniform_(self.state_map.weight, gain=1)
        torch.nn.init.constant_(self.state_map.bias, 0)
        self.state_map.train()

        # Update the features dim manually
        self._features_dim = 256 + 128

    def forward(self, observations) -> torch.Tensor:
        gripper_pos = observations["gripper_pos"]  # (N, 2)
        ee_pose_base = observations["tcp_pose"]  # (N, 6)
        action = observations["action"]  # (N, 7)
        grasp_exist = observations["grasp_exist"]  # (N, 5)
        grasps_posrot_ee = observations["grasps_posrot_ee"]  # (N, ng, 9)

        ## grasp_map to get 256 feat
        pn_feature, _, _ = self.pn(
            grasps_posrot_ee.transpose(2, 1).contiguous()
        )  # (N, 256)
        ## other state
        state = torch.cat([gripper_pos, ee_pose_base, action, grasp_exist], dim=1)
        state_feature = self.state_map(state)

        return torch.cat((state_feature, pn_feature), dim=1)
