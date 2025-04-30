from typing import List
import argparse
from collections import deque, defaultdict
from copy import deepcopy
import time
import itertools
import json

import numpy as np
import pickle
import torch
from pytorch3d.ops import ball_query, sample_farthest_points
from matplotlib import pyplot as plt
import open3d as o3d
import rospy
from tf.transformations import euler_from_quaternion

import gym
import h5py
from scipy.spatial.transform import Rotation, Slerp
from tqdm import tqdm

from stable_baselines3 import SAC

import sapien.core as sapien
from sapien.core import Pose

from gap_rl import ALGORITHM_DIR
from gap_rl.utils.common import inv_clip_and_scale_action, clip_and_scale_action
from gap_rl.utils.geometry import transform_points, qmul
from gap_rl.utils.wrappers.observation import DictObservationStack
from gap_rl.utils.sapien_utils import get_entity_by_name
from gap_rl.utils.wrappers.common import NormalizeBoxActionWrapper
from gap_rl.utils.o3d_utils import draw_o3d_geometries, crop_pcd

from gap_rl.localgrasp.LoG import LgNet, lg_parse, GraspGroup
from gap_rl.sim2real.robot_control import RobotUR
from gap_rl.sim2real.gripper_control import GripperController
from gap_rl.sim2real.config import pose_realsense_gripper
from gap_rl.sim2real.image_helpers import RealsenseCamera


def smooth_actions(actions: np.ndarray):
    # action size: (N, 7)
    steps = actions.shape[0]
    action_cache = deque(maxlen=3)

    act_pos, act_rot = [], []
    for step in range(steps):
        action_cache.append(actions[step])
        cache_size = len(action_cache)
        smooth_pos = 0
        rot_mat = np.eye(3)
        for ind in range(cache_size):
            smooth_pos += action_cache[ind][:3]
            delta_rot = action_cache[ind][3:6]
            rot_mat = rot_mat @ Rotation.from_rotvec(delta_rot).as_matrix()
        smooth_pos /= cache_size
        # smooth_rot = Rotation.from_matrix(rot_mat).as_rotvec() / cache_size
        slerp = Slerp([0, 1], Rotation.from_matrix([np.eye(3), rot_mat]))
        smooth_rot = slerp(np.linspace(0, 1, cache_size + 1)).as_rotvec()[1]
        act_pos.append(smooth_pos)
        act_rot.append(smooth_rot)
    return np.array(act_pos), np.array(act_rot), actions[:, -1]


def post_smoothtwists(twsits, smooth_mode='slidingwindow'):
    assert smooth_mode in ['slidingwindow', 'ema']
    smooth_twists = twsits.copy()
    traj_len = smooth_twists.shape[0]
    if smooth_mode == 'slidingwindow':
        window_len = 3
        for i in range(window_len, traj_len):
            smooth_twists[i] = np.mean(smooth_twists[i - window_len + 1:i + 1], axis=0)
        smooth_twists[0] = 0
    elif smooth_mode == 'ema':
        beta = 0.5
        for i in range(1, traj_len):
            smooth_twists[i] = beta * smooth_twists[i - 1] + (1 - beta) * smooth_twists[i]
    return smooth_twists


def fuse_pointcloud(pcds: List[np.ndarray], extrinsics: List[np.ndarray]):
    """
    pcds: list of pointclouds, shape [K, N, d]
    extrinsics: transformation matrix, shape [K, 4, 4]
    return: fused pointcloud, shape [K * N, d]
    """
    k = len(pcds)
    pcd_pts = [transform_points(extrinsics[id], pcds[id][:, :3]) for id in range(k)]
    if pcds[0].shape[1] > 3:
        pcd_feats = [pcds[i][:, 3:] for i in range(k)]
        pcd = np.concatenate((np.vstack(pcd_pts), np.vstack(pcd_feats)), axis=1)
    else:
        pcd = np.vstack(pcd_pts)
    return np.vstack(pcd)


def pointcloud_filter(points, *xyz_min_max, cuda=False):
    """
    :para points: [N, 3 + K]
    :para xyz_min_max: [3, 2], min and max of x, y, z
    """
    
    if cuda:
        xyz_min_max = xyz_min_max[0]
        mask1 = torch.logical_and(points[:, 0] > xyz_min_max[0][0], points[:, 0] < xyz_min_max[0][1])
        mask2 = torch.logical_and(points[:, 1] > xyz_min_max[1][0], points[:, 1] < xyz_min_max[1][1])
        mask3 = torch.logical_and(points[:, 2] > xyz_min_max[2][0], points[:, 2] < xyz_min_max[2][1])
        mask = torch.logical_and(torch.logical_and(mask1, mask2), mask3)
    else:
        xyz_min_max = xyz_min_max[0]
        mask1 = np.logical_and(points[:, 0] > xyz_min_max[0][0], points[:, 0] < xyz_min_max[0][1])
        mask2 = np.logical_and(points[:, 1] > xyz_min_max[1][0], points[:, 1] < xyz_min_max[1][1])
        mask3 = np.logical_and(points[:, 2] > xyz_min_max[2][0], points[:, 2] < xyz_min_max[2][1])
        mask = np.logical_and(np.logical_and(mask1, mask2), mask3)
    filtered_points = points[mask]
    return filtered_points, mask


def filter_grasp(world_gg, ee_gg, ground_offset=0.05, z_rot_offset=np.pi / 3):
    mask_pos = world_gg.translations[:, 2] > ground_offset
    mask_rot = ee_gg.rotations[:, 2, 2] > np.cos(z_rot_offset)
    filter_mask = mask_pos & mask_rot
    filter_ids = np.arange(world_gg.size)[filter_mask]
    return filter_ids


def compute_near_grasps_rt(grasps_mat_ee, grasps_scores, num_grasps=40):
    grasp_ids = np.arange(grasps_mat_ee.shape[0])  # (N, 4, 4)

    if len(grasp_ids) > num_grasps:
        grasp_ids = np.random.choice(grasp_ids, num_grasps, replace=False)

    grasps_ee = np.zeros((num_grasps, 4, 4))
    valid_grasp_num = len(grasp_ids)
    grasps_ee[:valid_grasp_num] = grasps_mat_ee[grasp_ids]

    filter_grasps_scores = np.zeros(num_grasps)
    filter_grasps_scores[:valid_grasp_num] = grasps_scores[grasp_ids]

    return grasps_ee, filter_grasps_scores


def pose_interp(pose1, pose2, interp_th=0.05):
    # print(pose1.q, pose2.q)
    p1, p2 = pose1.p, pose2.p
    q1, q2 = np.roll(pose1.q, -1), np.roll(pose2.q, -1) # wxyz - > xyzw
    # q1, q2 = pose1.q, pose2.q
    interp_steps_pos = int(np.linalg.norm(p1 - p2) / interp_th) + 1
    interp_steps_rot = max(5, interp_steps_pos)
    print('interp_steps', interp_steps_pos)
    if interp_steps_pos == 1:
        interp_pos = np.linspace(p1, p2, interp_steps_pos)[0] - p1
    else:
        interp_pos = np.linspace(p1, p2, interp_steps_pos)[1] - p1

    slerp = Slerp([0, 1], Rotation.from_quat([q1, q2]))
    cur_rot_base = Rotation.from_quat(q1).as_matrix()
    interp_rot_base = slerp(np.linspace(0, 1, interp_steps_rot)).as_matrix()[1]
    interp_rot = Rotation.from_matrix(interp_rot_base @ cur_rot_base.T).as_euler('XYZ')

    return interp_pos, interp_rot, interp_steps_pos


def pose_interp_steps(pose1, pose2, interp_steps=3):
    p1, p2 = pose1.p, pose2.p
    q1, q2 = np.roll(pose1.q, -1), np.roll(pose2.q, -1)  # wxyz - > xyzw
    if interp_steps == 1:
        interp_pos = np.linspace(p1, p2, interp_steps)[0] - p1
    else:
        interp_pos = np.linspace(p1, p2, interp_steps)[1] - p1

    slerp = Slerp([0, 1], Rotation.from_quat([q1, q2]))
    cur_rot_base = Rotation.from_quat(q1).as_matrix()
    interp_rot_base = slerp(np.linspace(0, 1, interp_steps)).as_matrix()[1]
    interp_rot = Rotation.from_matrix(interp_rot_base @ cur_rot_base.T).as_euler('XYZ')

    return interp_pos, interp_rot


def grasp_pose_dist(grasp_mat0, grasp_mats, trans_rot_ratio=1.0):
    tran0, trans = grasp_mat0[:3, 3], grasp_mats[:, :3, 3]
    translation_dist = np.linalg.norm(trans - tran0, axis=1)  # (N,)
    quat0 = np.roll(Rotation.from_matrix(grasp_mat0[:3, :3]).as_quat(), 1)
    quats = np.roll(Rotation.from_matrix(grasp_mats[:, :3, :3]).as_quat(), 1, axis=1)  # (N, 4) wxyz
    rot_dist0 = 1 - np.clip(np.abs(np.sum(quats * quat0, axis=-1)), a_min=0, a_max=1)
    quat0_rotz180 = qmul(quat0, np.array([0., 0., 0., 1.]))
    rot_dist1 = 1 - np.clip(np.abs(np.sum(quats * quat0_rotz180, axis=-1)), a_min=0, a_max=1)
    rotation_dist = np.minimum(rot_dist0, rot_dist1)  # (N,)
    grasp_dist = trans_rot_ratio * translation_dist + rotation_dist
    return grasp_dist, translation_dist, rotation_dist


def grasp_pose_ee_dist(grasp_mats_ee, trans_rot_ratio=1.0):
    translation_dist = np.linalg.norm(grasp_mats_ee[:, :3, 3], axis=1)  # (N,)
    quat = Rotation.from_matrix(grasp_mats_ee[:, :3, :3]).as_quat()  # (N, 4) xyzw
    rot_dist0 = 1 - np.clip(np.abs(quat[:, 2]), a_min=0, a_max=1)
    rot_dist1 = 1 - np.clip(np.abs(quat[:, 3]), a_min=0, a_max=1)
    rotation_dist = np.minimum(rot_dist0, rot_dist1)  # (N,)
    grasp_dist = trans_rot_ratio * translation_dist + rotation_dist
    reverse_flags = rot_dist0 < rot_dist1
    return grasp_dist, translation_dist, rotation_dist, reverse_flags


class Sim2Real:
    def __init__(self, args=None):
        self.args = args
        self.hand_cams = [
            # RealsenseCamera(camera_sn=self.args.realsense_sn[0]),
            RealsenseCamera(camera_sn=self.args.realsense_sn[1])
        ]
        self.handCam_to_ee_nps = [
            # sapien.Pose(p=pose_realsense[:3], q=pose_realsense[3:]).to_transformation_matrix(),
            sapien.Pose(p=pose_realsense_gripper[:3], q=pose_realsense_gripper[3:]).to_transformation_matrix()
        ]
        self.handCam_to_ees = [torch.from_numpy(self.handCam_to_ee_nps[i]).float().cuda() for i in range(1)]
        self.cam_id = 0
        self.num_grasps = 40

        if self.args.visualizer:
            self.vis_o3d = o3d.visualization.Visualizer()
            self.vis_o3d.create_window(window_name='sim2real', width=1920, height=1080, left=0, top=0)

        self.object_ws = deepcopy(self.args.ground_ws)
        self.object_ws[2][0] += self.args.object_z_offset  # base frame

        self.setup_robot(controller_type="twist_controller")
        print("robot setup")

        # Heuristic Plan param init
        self.object_vel = 0
        self.total_time = 1
        self.last_time = 0
        self.total_dist = 0
        self.last_obj_pos = 0
        self.final_velocity = 0
        self.step = 0

        if self.args.rl_mode in ["grasprt", "objpcrt"]:
            print("init rl model")
            self.rl_model = SAC.load(f"{self.args.model_path}/rl_model_2000000_steps", device="cuda:0")

        if self.args.rl_mode in ["grasprt", "heuristic_plan"]:
            print("init localgrasp")
            self.lgNet = LgNet(self.args)

        # RL-related param init
        self.action_bounds = [[-0.01, -0.01, -0.01, -0.05, -0.05, -0.05, 0],
                              [0.01, 0.01, 0.01, 0.05, 0.05, 0.05, 0.0425]]

    def test_cameras(self, trans_ee2base):
        hand_intrins = self.hand_cams[0].get_intrinsics_matrix()
        print('hand camera intrinsics: ', hand_intrins)
        # hand_rgb, hand_depth = self.hand_cams[0].get_image(color_depth=False)
        # plt.subplot(1, 2, 1)
        # plt.imshow((hand_rgb * 255.0).astype(np.uint8))
        # plt.subplot(1, 2, 2)
        # plt.imshow(visualize_depth(hand_depth))
        # plt.show()
        hand_points = self.hand_cams[0].get_pointcloud()
        print('hand camera points shape (hand_cam frame): ', hand_points.shape)
        draw_o3d_geometries([hand_points])
        # hand_rgbpc = np.concatenate((hand_points, hand_rgb.reshape(-1, 3)), axis=1)
        # draw_o3d_geometries([hand_rgbpc])
        handCam_to_base = trans_ee2base @ self.handCam_to_ee_nps[0]
        # rgbpc_base = transform_points(handCam_to_base, hand_rgbpc)
        # draw_o3d_geometries([rgbpc_base])
        pc_base = transform_points(handCam_to_base, hand_points)
        draw_o3d_geometries([pc_base])

        hand2_intrins = self.hand_cams[1].get_intrinsics_matrix()
        print('hand2 camera intrinsics: ', hand2_intrins)
        # hand2_rgb, hand2_depth = self.hand_cams[1].get_image(color_depth=False)
        # plt.subplot(1, 2, 1)
        # plt.imshow((hand2_rgb * 255.0).astype(np.uint8))
        # plt.subplot(1, 2, 2)
        # plt.imshow(visualize_depth(hand2_depth))
        # plt.show()
        hand2_points = self.hand_cams[1].get_pointcloud()
        print('hand camera points shape (hand_cam frame): ', hand2_points.shape)
        draw_o3d_geometries([hand2_points])
        # hand2_rgbpc = np.concatenate((hand2_points, hand2_rgb.reshape(-1, 3)), axis=1)
        # draw_o3d_geometries([hand2_rgbpc])
        handCam2_to_base = trans_ee2base @ self.handCam_to_ee_nps[1]
        # rgbpc2_base = transform_points(handCam2_to_base, hand2_rgbpc)
        # draw_o3d_geometries([rgbpc2_base])
        pc2_base = transform_points(handCam2_to_base, hand2_points)
        draw_o3d_geometries([pc2_base])

    def get_scene_points(self, trans_ee2base):
        hand_points = self.hand_cams[self.cam_id].get_pointcloud()
        print('hand camera points shape (hand_cam frame): ', hand_points.shape)
        if self.args.vis_data:
            draw_o3d_geometries([hand_points])

        handCam_to_base = trans_ee2base @ self.handCam_to_ee_nps[self.cam_id]
        pc_base = transform_points(handCam_to_base, hand_points)
        if self.args.vis_data:
            draw_o3d_geometries([pc_base])

        ## workspace filter
        # scene_pc_base, mask = pointcloud_filter(pc_base, self.args.ground_ws, cuda=True)
        pcd_base = o3d.geometry.PointCloud()
        pcd_base.points = o3d.utility.Vector3dVector(pc_base)
        crop_pcd_base = crop_pcd(pcd_base, list(itertools.product(*self.args.ground_ws)))
        if self.args.vis_data:
            draw_o3d_geometries([crop_pcd_base])
        scene_pc_base = np.asarray(crop_pcd_base.points)

        ## plane filter
        # obj_pcd_base, plane_cloud_base = remove_plane(crop_pcd_base, dist_th=0.03, ransac_n=3, num_it=1000)
        # obj_pc_base = np.asarray(obj_pcd_base.points)
        obj_pcd_base = crop_pcd(crop_pcd_base, list(itertools.product(*self.object_ws)))
        obj_pc_base = np.asarray(obj_pcd_base.points)
        if self.args.vis_data:
            # draw_o3d_geometries([plane_cloud_base])
            draw_o3d_geometries([obj_pcd_base])

        trans_base2ee = np.linalg.inv(trans_ee2base)
        scene_pc_ee, obj_pc_ee = transform_points(trans_base2ee, scene_pc_base), transform_points(trans_base2ee, obj_pc_base)

        for key in ["trans_ee2base", "scene_pc_base", "obj_pc_base", "scene_pc_ee", "obj_pc_ee"]:
            self.exp_info[key].append(eval(key))
        return scene_pc_ee, obj_pc_ee

    def setup_robot(self, controller_type="twist_controller"):
        rospy.init_node('ur_controller', anonymous=True)
        gripper_controller = GripperController(GRIPPER_PORT="/dev/ttyUSB0", sync=True)
        self.robot = RobotUR(gripper_controller, controller_type=controller_type)
        # self.robot.execute_combined_command(arm_action=HOME_JOINT_HIGHER, gripper_action=0.085)
        self.robot.switch_controller(controller_type=controller_type)

    def gen_grasps(self, trans_ee2base):
        t = time.time()
        assert self.lgNet is not None
        hand_points = torch.from_numpy(self.hand_cams[self.cam_id].get_pointcloud()).cuda()

        # record time
        if self.step <= 1:  # not compute for the initial two frame
            self.last_time = time.time()
        else:
            per_cost_time = time.time() - self.last_time
            self.total_time += per_cost_time
            self.last_time = time.time()

        # print('hand points shape:', hand_points.shape)
        print("get point cloud time: ", time.time() - t)
        t = time.time()
        handCam_to_base = torch.from_numpy(trans_ee2base).float().cuda() @ self.handCam_to_ees[self.cam_id]
        pc_base = transform_points(handCam_to_base, hand_points)
        print('compute_handCam_extrinsics time', time.time() - t)

        t = time.time()
        scene_rgbpc_base, mask = pointcloud_filter(pc_base, self.args.ground_ws, cuda=True)
        print('filtered scene points shape:', scene_rgbpc_base.shape)
        if self.args.vis_data:
            draw_o3d_geometries([scene_rgbpc_base.cpu().numpy()])
        scene_points_base = scene_rgbpc_base[:, :3]
        obj_z_th = self.object_ws[2][0]
        obj_rgbpc_base = scene_rgbpc_base[scene_rgbpc_base[:, 2] > obj_z_th]
        print('object points shape:', scene_rgbpc_base.shape)
        if self.args.vis_data:
            draw_o3d_geometries([obj_rgbpc_base.cpu().numpy()])

        if obj_rgbpc_base.shape[0] < 64:
            print('too few object points')
            return torch.zeros((0, 6)), torch.zeros((0, 6)), GraspGroup()

        obj_points_base = obj_rgbpc_base[:, :3]

        # record dist
        if self.step <= 1:  # not compute for the initial two frame
            self.last_obj_pos = obj_points_base.mean(0)
        else:
            per_dist = obj_points_base.mean(0) - self.last_obj_pos
            self.total_dist += per_dist
            print("****** cur velocity ****** : ", per_dist / per_cost_time)
            self.last_obj_pos = obj_points_base.mean(0)

        # obj_points = self.filter_pts(ground_filter_points, filter_workspace=False, filter_agent=True)
        # print('filtered obj points shape:', obj_rgbpc_base.shape)
        # if self.args.vis_data:
        #     draw_o3d_geometries([obj_rgbpc_base])

        # points: world => EE frame, generate grasps
        trans_ee2base_tensor = torch.from_numpy(trans_ee2base).float().cuda()
        trans_base2ee = torch.linalg.inv(trans_ee2base_tensor)
        scene_points_ee, obj_points_ee = transform_points(trans_base2ee, scene_points_base), transform_points(trans_base2ee, obj_points_base)
        print("filter point cloud time: ", time.time() - t)

        t = time.time()
        pred_gg_ee = self.lgNet.inference(
            obj_points=obj_points_ee,
            scene_points=scene_points_ee,
            num_grasps=64,
            scale=1.0
        )
        # print('nms filtered grasps number:', pred_gg_ee.size)

        # visualize the grasps
        # predgg_ee_homo = homo_transfer(
        #     R=pred_gg_ee.rotations, T=pred_gg_ee.translations
        # )
        # draw_gg_base = deepcopy(pred_gg_ee)
        # draw_graspmat_base = np.einsum('ij, kjl -> kil', trans_ee2base, predgg_ee_homo)  # (N, 4, 4)
        # draw_gg_base.translations = draw_graspmat_base[:, :3, 3]
        # draw_gg_base.rotations = draw_graspmat_base[:, :3, :3]

        if self.args.vis_data:
            draw_o3d_geometries([scene_points_ee.cpu().numpy()] + pred_gg_ee.to_open3d_geometry_list())

        self.grasp_ee.append(pred_gg_ee)
        self.obj_pc_ee.append(obj_points_ee.cpu().numpy())
        self.scene_pc_ee.append(scene_points_ee.cpu().numpy())

        # grasp pose representation (graspnet -> sapien)
        pred_gg_ee.rotations = np.einsum('ijk, kl -> ijl', pred_gg_ee.rotations, np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]))  # (N, 4, 4)
        print("grasp inference time: ", time.time() - t)

        return scene_rgbpc_base, obj_rgbpc_base, pred_gg_ee

    def init_grasps(self, trans_ee2base, vis=False):
        trans_ee2base_tensor = torch.from_numpy(trans_ee2base).float().cuda()
        t = time.time()
        assert self.lgNet is not None
        if self.args.save_rgb:
            hand_points, hand_rgb, hand_depth = self.hand_cams[self.cam_id].get_image_pointcloud(color_depth=False)
            hand_points_tensor = torch.from_numpy(hand_points).cuda()
        else:
            hand_points_tensor = torch.from_numpy(self.hand_cams[self.cam_id].get_pointcloud()).cuda()

        # print('hand points shape:', hand_points.shape)
        # print("get point cloud time: ", time.time() - t)
        t = time.time()
        handCam_to_base = trans_ee2base_tensor @ self.handCam_to_ees[self.cam_id]
        pc_base = transform_points(handCam_to_base, hand_points_tensor)
        # print('compute_handCam_extrinsics time', time.time() - t)

        t = time.time()
        scene_rgbpc_base, mask = pointcloud_filter(pc_base, self.args.ground_ws, cuda=True)
        # print('filtered scene points shape:', scene_rgbpc_base.shape)
        if vis:
            draw_o3d_geometries([scene_rgbpc_base.cpu().numpy()])
        scene_points_base = scene_rgbpc_base[:, :3]
        obj_z_th = self.object_ws[2][0]
        obj_rgbpc_base = scene_rgbpc_base[scene_rgbpc_base[:, 2] > obj_z_th]
        # print('object points shape:', scene_rgbpc_base.shape)
        if vis:
            draw_o3d_geometries([obj_rgbpc_base.cpu().numpy()])

        if obj_rgbpc_base.shape[0] < 64:
            print('too few object points')
            return torch.zeros((0, 6)), torch.zeros((0, 6)), GraspGroup()

        obj_points_base = obj_rgbpc_base[:, :3]

        # obj_points = self.filter_pts(ground_filter_points, filter_workspace=False, filter_agent=True)
        # print('filtered obj points shape:', obj_rgbpc_base.shape)
        # if self.args.vis_data:
        #     draw_o3d_geometries([obj_rgbpc_base])

        # points: world => EE frame, generate grasps
        trans_base2ee = torch.linalg.inv(trans_ee2base_tensor)
        scene_points_ee, obj_points_ee = transform_points(trans_base2ee, scene_points_base), transform_points(trans_base2ee, obj_points_base)
        # print("filter point cloud time: ", time.time() - t)

        t = time.time()
        centers_ee = sample_farthest_points(obj_points_ee[None], K=84)[0].squeeze()

        print("== ", centers_ee.shape)
        ## cluster centers to remove noisy centers
        centers_pcd = o3d.geometry.PointCloud()
        centers_pcd.points = o3d.utility.Vector3dVector(centers_ee.cpu().numpy())
        cluster_labels = np.array(centers_pcd.cluster_dbscan(eps=0.03, min_points=5, print_progress=False))
        centers_ee = centers_ee[cluster_labels == 0]
        print("== ", centers_ee.shape)
        print("FPS & DBCluster time: ", time.time() - t)

        centers_base = transform_points(trans_ee2base_tensor, centers_ee)
        pred_gg_ee = self.lgNet.infer_from_centers(scene_points_ee, centers_ee)
        # print('nms filtered grasps number:', pred_gg_ee.size)

        # visualize the grasps
        # predgg_ee_homo = homo_transfer(
        #     R=pred_gg_ee.rotations, T=pred_gg_ee.translations
        # )
        # draw_gg_base = deepcopy(pred_gg_ee)
        # draw_graspmat_base = np.einsum('ij, kjl -> kil', trans_ee2base, predgg_ee_homo)  # (N, 4, 4)
        # draw_gg_base.translations = draw_graspmat_base[:, :3, 3]
        # draw_gg_base.rotations = draw_graspmat_base[:, :3, :3]

        scene_points_ee, obj_points_ee = scene_points_ee.cpu().numpy(), obj_points_ee.cpu().numpy()
        pc_base, obj_points_base = pc_base.cpu().numpy(), obj_points_base.cpu().numpy()
        if vis:
            draw_o3d_geometries([scene_points_ee] + pred_gg_ee.to_open3d_geometry_list())

        # grasp pose representation (graspnet -> sapien)
        pred_gg_ee.rotations = np.einsum('ijk, kl -> ijl', pred_gg_ee.rotations, np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]))  # (N, 4, 4)
        # pred_gg_ee.rotations = np.einsum('ijk, kl -> ijl', pred_gg_ee.rotations, np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]]))  # (N, 4, 4)

        if self.args.save_rgb:
            return pc_base, obj_points_base, centers_base.cpu().numpy(), scene_points_ee, obj_points_ee, centers_ee.cpu().numpy(), pred_gg_ee, hand_rgb, hand_depth
        else:
            return pc_base, obj_points_base, centers_base.cpu().numpy(), scene_points_ee, obj_points_ee, centers_ee.cpu().numpy(), pred_gg_ee


    def track_grasps(self, prev_gg_ee, trans_ee2base):
        t = time.time()


        # Algorithm require: P_c^t-1 획득
        # ee 좌표계 -> 로봇 base 좌표계
        trans_ee2base_tensor = torch.from_numpy(trans_ee2base).float().cuda()
        # base -> ee
        trans_base2ee_tensor = torch.linalg.inv(trans_ee2base_tensor)
        # camera -> ee
        trans_hand2ee_tensor = self.handCam_to_ees[self.cam_id]

        # 이전 grasp 결과: EE좌표계 -> base 좌표계
        prev_gg_trans_base = transform_points(trans_ee2base_tensor, torch.from_numpy(prev_gg_ee.translations).float().cuda())
        # z축 높이로 바닥 필터링
        prev_gg_trans_base = prev_gg_trans_base[prev_gg_trans_base[:, 2] > self.object_ws[2][0]]
        # 서로 거리가 먼 점들을 K개 샘플링 => prev center라고 여김 (P_c^t-1)
        # TODO: 여기가 헷갈리네. 왜 previous center를 "sample" 하지?
        prev_gg_centers_base_tensor = sample_farthest_points(prev_gg_trans_base[None], K=self.args.num_centers[0])[0].squeeze()
        # prev_gg_centers_ee = transform_points(trans_base2ee_tensor, prev_gg_centers_base_tensor)
        print("get centers time: ", time.time() - t)


        # Algorithm require: 카메라 포인트 클라우드 획득 (P^t)
        t = time.time()
        if self.args.save_rgb:
            hand_points, hand_rgb, hand_depth = self.hand_cams[self.cam_id].get_image_pointcloud(color_depth=False)
        else:
            hand_points = self.hand_cams[self.cam_id].get_pointcloud()
        hand_points_tensor = torch.from_numpy(hand_points).float().cuda()
        # print('hand points shape:', hand_points.shape)
        print("get point cloud time: ", time.time() - t)


        t = time.time()
        # P^t: 카메라좌표계 => EE좌표계
        scene_points_ee_tensor = transform_points(trans_hand2ee_tensor, hand_points_tensor)
        # P^t: EE좌표계 => base좌표계
        scene_points_base_tensor = transform_points(trans_ee2base_tensor, scene_points_ee_tensor)
        scene_points_base = scene_points_base_tensor.cpu().numpy()


        ## filtered object points
        # 작업범위 아닌곳 필터링
        scene_points_base_tensor, mask = pointcloud_filter(scene_points_base_tensor, self.args.ground_ws, cuda=True)


        # Algorithm line 5
        # line6을 보면, center 후보군은 (1) t초의 PCL 중 이전 grasp 위치에서 가까운 point들을 중 N_o개 샘플링 +
        # (2) t초의 PCL 에서 그냥 N_c개 샘플링 (얘가 더 넓은 범위에서 샘플링)
        # 밑의 add_centers_base_tensor가 N_c개 샘플링하는 과정
        # 추가적으로 더 좁은 범위인 물체(object)가 있을 것으로 예상되는 공간 필터링
        object_points_base_tensor, mask = pointcloud_filter(scene_points_base_tensor, self.object_ws, cuda=True)
        # object_points: base 좌표 -> EE 좌표
        object_points_ee_tensor = transform_points(trans_base2ee_tensor, object_points_base_tensor)
        obj_points_base, obj_points_ee = object_points_base_tensor.cpu().numpy(), object_points_ee_tensor.cpu().numpy()
        # object_pionts 중에서 FPS로 K개 샘플링 -> 아마 complement center로 활용하는듯? (gpt)
        add_centers_base_tensor = sample_farthest_points(object_points_base_tensor[None], K=self.args.num_centers[1])[0].squeeze()  # (nc, 3)
        # add_centers_ee_tensor = transform_points(trans_base2ee_tensor, add_centers_base_tensor)
        print("filtered time: ", time.time() - t)


        # Algorithm line 2
        # Ball query => p1을 중심으로 원을 그림
        # 그 공 안에 있는 포인트(p2의 점)를 K개 찾음.
        # 즉, 이전 grasp의 center 주변의 '현재 프레임 포인트 클라우드'를 찾는다.
        # 즉, t-1의 graspable한 위치에서 멀리 떨어지지 않은, t의 PCL을 찾는다.
        # Summary: Input:p1=P_c^t-1, p2=P^t (line 2) // Output: P_i^t
        t = time.time()
        ## nk ball_query points group => fps(64)
        print(prev_gg_centers_base_tensor.shape, scene_points_base_tensor.shape)
        dists, idx, nn = ball_query(
            p1=prev_gg_centers_base_tensor[None],
            p2=scene_points_base_tensor[None],
            K=256,
            radius=self.args.ball_query_r,
        )
        dists, idx, nn = dists[0], idx[0], nn[0]
        ii, jj = torch.where(dists != 0)
        track_region_points_base_tensor = nn[ii, jj]  # (M, 3)
        print("origin shape: ", track_region_points_base_tensor.shape)
        track_region_points_base_tensor = torch.unique(track_region_points_base_tensor, dim=0)
        print("processing shape: ", track_region_points_base_tensor.shape)


        # Algorithm line 3: ball_query의 output들 바닥이네 물체 workspace 아래에 존재하는 노이즈를 필터링한다
        ## ground filter
        g_filter = track_region_points_base_tensor[:, 2] > self.object_ws[2][0]
        track_region_points_base_tensor = track_region_points_base_tensor[g_filter]


        # Algorithm line 4: P_o^t 획득 (line 3의 filtered output들 중 graspable한 것들을 sampling한다)
        # FPS: 포인트 클라우드의 모양(shape)을 잘 나타내는 대표 지점들을 샘플링 (좋은 후보군들을 추출하는 기능)
        try:
            track_centers_base_tensor = sample_farthest_points(track_region_points_base_tensor[None], K=self.args.num_centers[0])[0].squeeze()  # (nc, 3)
        except:
            breakpoint()
        print("BQ & FPS time: ", time.time() - t)


        t = time.time()
        # track_centers_ee_tensor = transform_points(trans_base2ee_tensor, track_centers_base_tensor)


        # Algorithm line 6: P^t_complete와 P^t_target concat
        update_centers_base_tensor = torch.cat((add_centers_base_tensor, track_centers_base_tensor), dim=0)
        print("== ", update_centers_base_tensor.shape)


        # 아마도 algorithm line 7. clustering 알고리즘으로, 가장 큰 cluster (cluster_labels == 0)만 남김. 즉 noise 제거의 효과.
        ## cluster centers to remove noisy centers
        centers_pcd = o3d.geometry.PointCloud()
        centers_pcd.points = o3d.utility.Vector3dVector(update_centers_base_tensor.cpu().numpy())
        cluster_labels = np.array(centers_pcd.cluster_dbscan(eps=0.03, min_points=5, print_progress=False))
        update_centers_base_tensor = update_centers_base_tensor[cluster_labels == 0]
        print("== ", update_centers_base_tensor.shape)
        print("DBCluster time: ", time.time() - t)




        ## filter outliers
        # dists, idx, nn = ball_query(
        #     p1=update_centers_base_tensor[None],
        #     p2=update_centers_base_tensor[None],
        #     K=12,
        #     radius=self.args.ball_query_r,
        # )
        # update_centers_base_tensor = update_centers_base_tensor[((dists[0] > 0).sum(dim=1)) > 11]

        # update_centers_ee_tensor = torch.cat((add_centers_ee_tensor, track_centers_ee_tensor), dim=0)


        # 위는 다 base frame에서 계산함. 실제 grasp 수행 시 EE frame로 할거라, base->ee 변환 해야함.
        update_centers_ee_tensor = transform_points(trans_base2ee_tensor, update_centers_base_tensor)

        scene_points_ee = scene_points_ee_tensor.cpu().numpy()
        update_centers_ee = update_centers_ee_tensor.cpu().numpy()
        update_centers_base = update_centers_base_tensor.cpu().numpy()

        if self.args.vis_data:
            g_cs = []
            for i in range(sum(self.args.num_centers)):
                m = o3d.geometry.TriangleMesh.create_sphere(0.01)
                m.translate(update_centers_ee_tensor.cpu().numpy()[i])
                m.paint_uniform_color([0, 1, 0])
                g_cs.append(m)
            draw_o3d_geometries([scene_points_ee] + g_cs)  # points_ee + grasp centers

        # print("center computation time: ", time.time() - t)
        t = time.time()


        # Section 4-C: grasp detection. 
        # Use Local Grasp 모델. 
        # scene_points: P^t (ee frame): EE 입장에서 본 scene
        # centers: \hat{P_c^t} (ee frame): 위 과정에서 구한 grasp 위치 후보
        # pred_gg_ee => 후보 각각의 위치 (translation), 방향 (rotation), 품질 점수 (scores) 등을 나타냄
        # https://arxiv.org/pdf/2403.15054v1 논문을 잠깐 봤는데도, 좀 헷갈리네
        pred_gg_ee = self.lgNet.infer_from_centers(
            scene_points=scene_points_ee_tensor,
            centers=update_centers_ee_tensor
        )

        # print('nms filtered grasps number:', pred_gg_ee.size)

        # visualize the grasps
        # predgg_ee_homo = homo_transfer(
        #     R=pred_gg_ee.rotations, T=pred_gg_ee.translations
        # )
        # draw_gg_base = deepcopy(pred_gg_ee)
        # draw_graspmat_base = np.einsum('ij, kjl -> kil', trans_ee2base, predgg_ee_homo)  # (N, 4, 4)
        # draw_gg_base.translations = draw_graspmat_base[:, :3, 3]
        # draw_gg_base.rotations = draw_graspmat_base[:, :3, :3]

        if self.args.vis_data:
            draw_o3d_geometries([scene_points_ee] + pred_gg_ee.to_open3d_geometry_list())

        # Local grasp의 output은 graspnet 좌표계. 이걸 sapien 좌표계로 바꿔준다. (gpt: (x,y,z) => (z,x,y))
        # grasp pose representation (graspnet -> sapien)
        pred_gg_ee.rotations = np.einsum('ijk, kl -> ijl', pred_gg_ee.rotations, np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]))  # (N, 4, 4)
        print("grasp inference time: ", time.time() - t)

        # return scene_points_base, update_centers_base, scene_points_ee, update_centers_ee, pred_gg_ee, hand_rgb, hand_depth
        if self.args.save_rgb:
            return scene_points_base, obj_points_base, update_centers_base, scene_points_ee, obj_points_ee, update_centers_ee, pred_gg_ee, hand_rgb, hand_depth
        else:
            return scene_points_base, obj_points_base, update_centers_base, scene_points_ee, obj_points_ee, update_centers_ee, pred_gg_ee

    def test_ours(self, ):
        trans_base2world = np.eye(4)
        trans_base2world[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        gripper_pts_ee = np.random.normal(0.0, 0.0425 / 3, size=(20, 3))
        control_freq = 5
        step = 0
        real_action = np.zeros(7)
        min_grasp_dist = np.inf
        self.exp_info = defaultdict(list)

        with torch.no_grad():
            ## get robot states
            robot_state_0 = self.robot.get_real_state()
            tcp_pose = robot_state_0['tcp_state']  # (pos, wxyz), base frame
            cur_pose_0 = Pose(p=tcp_pose[:3], q=tcp_pose[3:])
            cur_ee_mat_world = trans_base2world @ Pose(p=tcp_pose[:3], q=tcp_pose[3:]).to_transformation_matrix()
            cur_ee_euler = Rotation.from_matrix(cur_ee_mat_world[:3, :3]).as_euler("XYZ")
            cur_obs = {
                # :para gripper_pos: [0, 0.085], close -> open
                'gripper_pos': np.repeat((1 - robot_state_0['gripper_pos'] / 0.085) * 0.0425, 2).astype(np.float32),  # [0, 0.0425]
                'tcp_pose': np.hstack([cur_ee_mat_world[:3, 3], cur_ee_euler]).astype(np.float32),
                # 'grasp_state': np.repeat(robot_state_0['grasp_state'], 5),
                'action': real_action.astype(np.float32),
            }
            trans_ee2base = cur_pose_0.to_transformation_matrix()
            if self.args.save_rgb:
                scene_points_base, obj_points_base, centers_base, scene_points_ee, obj_points_ee, centers_ee, pred_gg_ee, hand_rgb, hand_depth = self.init_grasps(
                    trans_ee2base=trans_ee2base,
                )
            else:
                scene_points_base, obj_points_base, centers_base, scene_points_ee, obj_points_ee, centers_ee, pred_gg_ee = self.init_grasps(
                    trans_ee2base=trans_ee2base,
                )
            grasps_num = pred_gg_ee.size
            for key in ["trans_ee2base", "scene_points_base", "obj_points_base", "centers_base", "scene_points_ee", "obj_points_ee", "centers_ee", "pred_gg_ee"]:
                self.exp_info[key].append(eval(key))
            if self.args.save_rgb:
                for key in ["hand_rgb", "hand_depth"]:
                    self.exp_info[key].append(eval(key))

            min_grasp_ids = []
            # min_trans_dist, min_rot_dist = np.inf, np.inf
            while len(min_grasp_ids) == 0 and step < self.args.max_steps:
            # while min_grasp_dist > gg_tcp_dist_th and step < max_steps:
                start = time.time()
                ## grasp sampling & transform to 3d points
                grasps_ee_0 = np.repeat(np.eye(4)[None], pred_gg_ee.size, axis=0)  # (N, 4, 4)
                grasps_ee_0[:, :3, 3] = pred_gg_ee.translations
                grasps_ee_0[:, :3, :3] = pred_gg_ee.rotations
                grasps_ee, grasps_scores = compute_near_grasps_rt(grasps_ee_0, pred_gg_ee.scores, num_grasps=self.num_grasps)
                R, T = grasps_ee[:, :3, :3].transpose((0, 2, 1)), grasps_ee[:, :3, 3]
                gripper_pts_diff = np.einsum("ij, kjl -> kil", gripper_pts_ee, R) + np.repeat(
                    T[:, None, :], gripper_pts_ee.shape[0], axis=1
                )  # (num, k, 3)
                grasp_exist = np.ones(5) if grasps_num > 0 else np.zeros(5)
                cur_obs.update(
                    {'grasp_exist': grasp_exist.astype(np.float32),
                     'gripper_pts_diff': gripper_pts_diff.astype(np.float32),
                     'grasps_scores': grasps_scores.astype(np.float32),
                     'close_grasp_pose_ee': np.zeros(7).astype(np.float32)}
                )
                # print('grasp sampling & transform to 3d points, time', time.time() - start)

                ## RL prediction & grasp goal prediction
                t = time.time()
                action, _states = self.rl_model.predict(cur_obs, deterministic=True)
                real_action = clip_and_scale_action(action, self.action_bounds[0], self.action_bounds[1])
                obs_tensor = {k: torch.tensor(o[None], device="cuda:0") for k, o in cur_obs.items()}
                action_tensor = torch.tensor(action[None], device="cuda:0")
                _, pred_pose_actor = self.rl_model.actor.action_log_prob(obs_tensor)
                _, pred_pose_critic = self.rl_model.critic(obs_tensor, action_tensor)
                pred_grasp_actor_critic = torch.cat((pred_pose_actor, pred_pose_critic), dim=0).cpu().numpy()  # (2, 7)
                pred_grasp_ac_rotmat = Rotation.from_quat(np.roll(pred_grasp_actor_critic[:, :4], -1)).as_matrix()
                pred_grasp_ac = GraspGroup(
                    translations=pred_grasp_actor_critic[:, 4:],
                    rotations=pred_grasp_ac_rotmat @ np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]),
                    widths=np.array([0.085, 0.085]),
                    depths=np.array([0.02, 0.02]),
                    scores=np.array([1, 0])
                )
                self.exp_info["pred_grasps_ac"].append(pred_grasp_ac)
                # print('RL & grasp goal prediction: ', time.time() - t)

                ## save pn feat
                self.exp_info["actor_pnfeat"].append(self.rl_model.actor.features_forward(obs_tensor)[0, 128:].cpu().numpy())
                self.exp_info["critic_pnfeat"].append(self.rl_model.critic.features_forward(obs_tensor)[0, 128:].cpu().numpy())

                ## get real control command (action => twist in base frame)
                t = time.time()
                delta_pos, delta_euler, gripper_action = real_action[0:3], real_action[3:6], real_action[6]
                delta_quat = np.roll(Rotation.from_euler('XYZ', delta_euler).as_quat(), 1)
                print("delta pos, delta euler (align EE frame): ", delta_pos, delta_euler)
                target_pose_0 = cur_pose_0 * sapien.Pose(p=delta_pos, q=delta_quat)
                twist_base = target_pose_0 * cur_pose_0.inv()
                vel, wvel = target_pose_0.p - cur_pose_0.p, euler_from_quaternion(np.roll(twist_base.q, -1))
                twist = np.concatenate((vel, wvel))
                real_twist = twist * control_freq
                # real_twist[3:] = 0
                gripper_cmd = 2 * (0.0425 - gripper_action)
                # print('action -> base twist, time: ', time.time() - t)
                self.exp_info["real_action"].append(np.append(real_twist, gripper_cmd))

                ## robot execution
                step += 1
                self.robot.execute_combined_command(real_twist, gripper_cmd)
                print('command executed!')

                ## get robot states
                robot_state_0 = self.robot.get_real_state()
                tcp_pose = robot_state_0['tcp_state']  # (pos, wxyz), base frame
                cur_pose_0 = Pose(p=tcp_pose[:3], q=tcp_pose[3:])
                cur_ee_mat_world = trans_base2world @ Pose(p=tcp_pose[:3], q=tcp_pose[3:]).to_transformation_matrix()
                cur_ee_euler = Rotation.from_matrix(cur_ee_mat_world[:3, :3]).as_euler("XYZ")
                cur_obs = {
                    # :para gripper_pos: [0, 0.085], close -> open
                    'gripper_pos': np.repeat((1 - robot_state_0['gripper_pos'] / 0.085) * 0.0425, 2).astype(np.float32),  # [0, 0.0425]
                    'tcp_pose': np.hstack([cur_ee_mat_world[:3, 3], cur_ee_euler]).astype(np.float32),
                    # 'grasp_state': np.repeat(robot_state_0['grasp_state'], 5),
                    'action': real_action.astype(np.float32),
                }
                trans_ee2base = cur_pose_0.to_transformation_matrix()

                t0 = time.time()
                if self.args.save_rgb:
                    if self.args.real_mode == "ws_filter":
                        ## generate grasp using filtered points
                        scene_points_base, obj_points_base, centers_base, scene_points_ee, obj_points_ee, centers_ee, pred_gg_ee, hand_rgb, hand_depth = self.init_grasps(
                            trans_ee2base=trans_ee2base,
                        )
                    elif self.args.real_mode == "explorer":
                        ## generate grasp using tracked grasp centers
                        scene_points_base, obj_points_base, centers_base, scene_points_ee, obj_points_ee, centers_ee, pred_gg_ee, hand_rgb, hand_depth = self.track_grasps(
                            prev_gg_ee=pred_gg_ee,
                            trans_ee2base=trans_ee2base,
                        )  # obj_points_ee == update_centers_ee
                    else:
                        raise NotImplementedError
                else:
                    if self.args.real_mode == "ws_filter":
                        ## generate grasp using filtered points
                        scene_points_base, obj_points_base, centers_base, scene_points_ee, obj_points_ee, centers_ee, pred_gg_ee = self.init_grasps(
                            trans_ee2base=trans_ee2base,
                        )
                    elif self.args.real_mode == "explorer":
                        ## generate grasp using tracked grasp centers
                        scene_points_base, obj_points_base, centers_base, scene_points_ee, obj_points_ee, centers_ee, pred_gg_ee = self.track_grasps(
                            prev_gg_ee=pred_gg_ee,
                            trans_ee2base=trans_ee2base,
                        )  # obj_points_ee == update_centers_ee
                    else:
                        raise NotImplementedError
                print("get grasp time: ", time.time() - t0)
                grasps_num = pred_gg_ee.size
                # if grasps_num > 0:
                # grasp_tcp_dist = np.linalg.norm(pred_gg_ee.translations)
                grasp_mats_ee = np.eye(4)[None].repeat(grasps_num, 0)
                grasp_mats_ee[:, :3, 3], grasp_mats_ee[:, :3, :3] = pred_gg_ee.translations, pred_gg_ee.rotations
                # grasp_tcp_dist, tran_dist, rot_dist = grasp_pose_dist(np.eye(4), grasp_mats_ee, trans_rot_ratio=1.0)
                grasp_dist, trans_dist, rot_dist, reverse_flags = grasp_pose_ee_dist(grasp_mats_ee, trans_rot_ratio=1.0)
                # print("==== total_dist: ", grasp_dist)
                # print("==== tran_dist: ", trans_dist)
                # print("==== rot_dist: ", rot_dist)
                # min_grasp_dist = grasp_dist.min()
                # min_grasp_id = int(np.argmin(grasp_dist))
                # print("==== min dist, min id: ", min_grasp_dist, min_grasp_id)
                grasp_ids = np.arange(grasps_num)
                trans_ids = set(grasp_ids[trans_dist < self.args.trans_dist_th])
                rot_ids = set(grasp_ids[rot_dist < self.args.rot_dist_th])
                min_grasp_ids = list(trans_ids & rot_ids)
                print("==== min ids: ", min_grasp_ids)
                if len(min_grasp_ids) > 0:
                    print("==== min trasn, rot dist: ", trans_dist[min_grasp_ids[0]], rot_dist[min_grasp_ids[0]])
                for key in ["trans_ee2base", "scene_points_base", "obj_points_base", "centers_base", "scene_points_ee", "obj_points_ee", "centers_ee", "pred_gg_ee"]:
                    self.exp_info[key].append(eval(key))
                if self.args.save_rgb:
                    for key in ["hand_rgb", "hand_depth"]:
                        self.exp_info[key].append(eval(key))

                print(f"step {step} ++++++++++", time.time() - start)

        # stop & open gripper
        min_grasp_id = min_grasp_ids[0]

        ## move to target_grasp
        robot_state_0 = self.robot.get_real_state()
        tcp_pose = robot_state_0['tcp_state']  # (pos, wxyz), base frame
        cur_pose_0 = Pose(p=tcp_pose[:3], q=tcp_pose[3:])
        trans_ee2base = cur_pose_0.to_transformation_matrix()

        final_grasp_mat_ee = grasp_mats_ee[min_grasp_id]
        if reverse_flags[min_grasp_id]:
            final_grasp_mat_ee[:3, :3] = final_grasp_mat_ee[:3, :3] @ np.diag([-1, -1, 1])
        final_grasp_ac = GraspGroup(
            translations=final_grasp_mat_ee[:3, 3][None],
            rotations=(final_grasp_mat_ee[:3, :3] @ np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]))[None],
            widths=np.array([0.085]),
            depths=np.array([0.02]),
            scores=np.array([1])
        )
        self.exp_info["pred_grasps_ac"].append(final_grasp_ac)
        final_grasp_base = Pose().from_transformation_matrix(trans_ee2base @ final_grasp_mat_ee)

        with open(f"/data/challenge_grasp_logs/real_exps/exp_info_{self.args.traj_mode}.pkl", "wb") as file:
            pickle.dump(self.exp_info, file)

        interp_steps = 2
        interp_pos, interp_euler = pose_interp_steps(cur_pose_0, final_grasp_base, interp_steps=interp_steps)  # in base frame
        for i in range(interp_steps):
            t = time.time()
            real_twist = np.concatenate((interp_pos, interp_euler)) * control_freq * self.args.speed_para
            self.robot.execute_combined_command(real_twist, 0.085)  # gripper: [0, 0.085], close -> open
            print("open-loop control time: ", time.time() - t)
        print('Move to final grasp pose!')

        ## close gripper & lift
        arm_gripper_cmds = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.03, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.085]
        ] * control_freq
        # close gripper
        self.robot.execute_combined_command(arm_gripper_cmds[0][:6], arm_gripper_cmds[0][6])
        time.sleep(1)
        # lift the object
        self.robot.execute_combined_command(arm_gripper_cmds[1][:6], arm_gripper_cmds[1][6])
        time.sleep(5)
        # stop
        self.robot.execute_combined_command(arm_gripper_cmds[2][:6], arm_gripper_cmds[2][6])
        time.sleep(1)
        # open gripper
        self.robot.execute_combined_command(arm_gripper_cmds[3][:6], arm_gripper_cmds[3][6])
        time.sleep(1)

        # vis_steps = [0, 1] + list(range(step-5, step))
        # vis_steps = list(range(1, step))
        # for i in vis_steps:
        #     print(f"step {i} ++++++++++")
        #     scene_pc_ee = self.exp_info["scene_points_ee"][i]
        #     obj_pc_ee = self.exp_info["obj_points_ee"][i]
        #     pred_grasp_ac = self.exp_info["pred_grasps_ac"][i]
        #     pred_gg_ee = self.exp_info["pred_gg_ee"][i]
        #     pred_gg_ee.rotations = np.einsum('ijk, kl -> ijl', pred_gg_ee.rotations, np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]))  # (N, 4, 4)
        #     centers_ee = self.exp_info["centers_ee"][i]
        #     ms = []
        #     centers_num = centers_ee.shape[0]
        #     for j in range(centers_num):
        #         m = o3d.geometry.TriangleMesh.create_sphere(0.003)
        #         m.translate(centers_ee[j])
        #         m.paint_uniform_color([0, 1, 0])
        #         ms.append(m)
        #     draw_o3d_geometries(
        #         [np.concatenate((obj_pc_ee + np.array([0, 0, -0.01]), np.array([[1, 0, 0]]).repeat(obj_pc_ee.shape[0], 0)), axis=1),
        #          np.concatenate((scene_pc_ee, np.array([[0, 0, 1]]).repeat(scene_pc_ee.shape[0], 0)), axis=1)] +
        #         pred_grasp_ac.to_open3d_geometry_list(size=4) +
        #         pred_gg_ee.to_open3d_geometry_list() +
        #         ms
        #     )

        # print(f"The last step ++++++++++")
        # scene_pc_ee = self.exp_info["scene_points_ee"][-1]
        # obj_pc_ee = self.exp_info["obj_points_ee"][-1]
        # final_grasp = self.exp_info["pred_grasps_ac"][-1]
        # pred_gg_ee = self.exp_info["pred_gg_ee"][-1]
        # pred_gg_ee.rotations = np.einsum('ijk, kl -> ijl', pred_gg_ee.rotations, np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]))  # (N, 4, 4)
        # centers_ee = self.exp_info["centers_ee"][-1]
        # ms = []
        # centers_num = centers_ee.shape[0]
        # for j in range(centers_num):
        #     m = o3d.geometry.TriangleMesh.create_sphere(0.003)
        #     m.translate(centers_ee[j])
        #     m.paint_uniform_color([0, 1, 0])
        #     ms.append(m)
        # draw_o3d_geometries(
        #     [np.concatenate((obj_pc_ee + np.array([0, 0, -0.01]), np.array([[1, 0, 0]]).repeat(obj_pc_ee.shape[0], 0)), axis=1),
        #      np.concatenate((scene_pc_ee, np.array([[0, 0, 1]]).repeat(scene_pc_ee.shape[0], 0)), axis=1)] +
        #     final_grasp.to_open3d_geometry_list(size=4) +
        #     pred_gg_ee.to_open3d_geometry_list() +
        #     ms
        # )
        print("Finish ! ! !")
        # self.vis_o3d.destroy_window()

    def test_heuristic_plan(self):
        if self.args.task == 'handover':
            control_freq = 5  # for hand over
            fix_grasp_dist = 0.10
        elif args.task == 'dynamic':
            control_freq = 10  # for dynamic grasp
            fix_grasp_dist = 0.15
        with torch.no_grad():
            grasp_ee_final = None
            for step in tqdm(range(args.max_steps), colour='red', leave=False):
                self.step = step
                start = time.time()

                # get grasps
                if isinstance(grasp_ee_final, np.ndarray):
                    grasps_ee_0 = grasp_ee_final[None]
                else:
                    t = time.time()
                    robot_state_0 = self.robot.get_real_state()
                    cur_pose_0 = Pose(p=robot_state_0['tcp_state'][:3], q=robot_state_0['tcp_state'][3:])
                    trans_ee2base = cur_pose_0.to_transformation_matrix()
                    scene_rgbpc_base, obj_rgbpc_base, pred_gg_ee = self.gen_grasps(trans_ee2base=trans_ee2base)
                    grasps_num = pred_gg_ee.size
                    print('whole generate grasp time: ', time.time() - t)
                    grasps_ee_0 = np.repeat(np.eye(4)[None], pred_gg_ee.size, axis=0)  # (N, 4, 4)
                    grasps_ee_0[:, :3, 3] = pred_gg_ee.translations
                    grasps_ee_0[:, :3, :3] = pred_gg_ee.rotations
                    R, T = grasps_ee_0[:, :3, :3].transpose((0, 2, 1)), grasps_ee_0[:, :3, 3]

                if args.method == 'simple_interp':
                    t = time.time()
                    robot_state_1 = self.robot.get_real_state()
                    print("get state time: ", time.time() - t)
                    cur_pose_1 = Pose(p=robot_state_1['tcp_state'][:3], q=robot_state_1['tcp_state'][3:])
                    trans_e0e1 = cur_pose_1.inv().to_transformation_matrix() @ cur_pose_0.to_transformation_matrix()  # e0 -> e1
                    grasps_ee = np.einsum('ij, kjl -> kil', trans_e0e1, grasps_ee_0)  # (N, 4, 4)
                    if isinstance(grasp_ee_final, np.ndarray):
                        translation_dist = np.linalg.norm(grasps_ee[:, 1:3, 3], axis=1)  # only consider (y,z) due to x velocity plus
                    else:
                        translation_dist = np.linalg.norm(grasps_ee[:, :3, 3], axis=1)  # (sn,)
                    # rot dist using quat inner product
                    quat = Rotation.from_matrix(grasps_ee[:, :3, :3]).as_quat()  # (sn, 4), xyzw
                    rot_dist0 = 1 - np.clip(np.abs(quat[:, 2]), a_min=0, a_max=1)
                    rot_dist1 = 1 - np.clip(np.abs(quat[:, 3]), a_min=0, a_max=1)
                    rotation_dist = np.minimum(rot_dist0, rot_dist1)
                    # reverse_flags = np.argmin([rot_dist0, rot_dist1], axis=0)
                    reverse_flags = rot_dist0 < rot_dist1
                    
                    # reverse_ids = np.arange(pred_gg_base.size)[reverse_flags]

                    grasp_dist = translation_dist + rotation_dist
                    grasp_id = np.argmin(grasp_dist)
                    # grasp_id = 0
                    grasp_ee_choose = grasps_ee[grasp_id]
                    if grasp_ee_choose[2, 3] < fix_grasp_dist:
                        print("fix grasp pose**********")
                        grasp_ee_final = grasp_ee_choose
                        robot_state_0 = robot_state_1
                        cur_pose_0 = cur_pose_1
                        self.final_velocity = self.total_dist / self.total_time
                        # print(grasp_ee_choose)
                        # print(grasp_ee_choose_addvelo)
                        print("*** final velocity ****", self.final_velocity)
                        
                    print('translation_dist: ', translation_dist[grasp_id])
                    print('rotation_dist: ', rotation_dist[grasp_id])
                    print('grasp_ee_choose', grasp_ee_choose)
                    # print('base frame, target_pos, cur_pos: ', pred_gg_base.translations[grasp_id], grasps_ee[:, :3, 3][grasp_id], cur_pose_1.p)
                    # print('min grasp_dist: ', grasp_dist[grasp_id], translation_dist[grasp_id], rotation_dist[grasp_id])
                    # draw_gg_base.scores = np.zeros(draw_gg_base.size)
                    # draw_gg_base.scores[grasp_id] = 1  # set the best grasp red
                    # draw_o3d_geometries([scene_rgbpc_base] + draw_gg_base.to_open3d_geometry_list())

                    # ctr = self.vis_o3d.get_view_control()
                    # ctr.set_front([-1, 0, -1])
                    # ctr.set_up([-1, 0, 1])
                    # ctr.set_lookat([0, 0, 0])
                    # # ctr.rotate(0, 350)
                    # # ctr.set_zoom(2)
                    # self.vis_o3d.clear_geometries()
                    # self.vis_o3d.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1))
                    # scene_rgbpc_o3d = o3d.geometry.PointCloud()
                    # scene_rgbpc_o3d.points = o3d.utility.Vector3dVector(scene_rgbpc_base[:, :3])
                    # scene_rgbpc_o3d.colors = o3d.utility.Vector3dVector(scene_rgbpc_base[:, 3:])
                    # self.vis_o3d.add_geometry(scene_rgbpc_o3d)
                    # if draw_gg_base.size > 0:
                    #     grasps_geo = draw_gg_base.to_open3d_geometry_list()
                    #     cur_grasp = Grasp(translation=cur_pose.p, rotation=Pose(q=cur_pose.q).to_transformation_matrix()[:3, :3] @ np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]))
                    #     grasps_geo.append(cur_grasp.to_open3d_geometry(color=[1, 1, 0]))
                    #     for g in grasps_geo:
                    #         self.vis_o3d.add_geometry(g)
                    # self.vis_o3d.poll_events()
                    # self.vis_o3d.update_renderer()
                    # time.sleep(0.01)

                    if reverse_flags[grasp_id]:
                        grasp_ee_choose[:3, :3] = grasp_ee_choose[:3, :3] @ np.diag([-1, -1, 1])
                        
                    # transform to base frame
                    trans_ee2base = cur_pose_1.to_transformation_matrix() # e0 -> e1
                    grasp_base_choose = trans_ee2base @ grasp_ee_choose
                    
                    if isinstance(grasp_ee_final, np.ndarray):
                        # if np.abs(self.final_velocity[0].cpu().numpy()) > 0.02:
                        grasp_base_choose[:1, 3] += np.sign(target_pose.p[0]) * 0.02

                    target_pose = Pose(
                        p=grasp_base_choose[:3, 3],
                        q=Rotation.from_matrix(grasp_base_choose[:3, :3]).as_quat()[[3, 0, 1, 2]]
                    )
                
                    interp_pos, interp_euler, interp_steps_pos = pose_interp(cur_pose_1, target_pose, interp_th=0.025) # in base frame
                    
                    # interp_pos = np.zeros(3)
                    rot_thre = 0.02
                    pos_thre = 0.02
                
                    if rotation_dist[grasp_id] < rot_thre and translation_dist[grasp_id] < pos_thre:
                        print(" !!! arrive grasp pose !!! ")
                        break
                    elif rotation_dist[grasp_id] < rot_thre:
                        print(" !!! arrive grasp rotation !!! ")
                        interp_euler = np.zeros(3)
                    elif translation_dist[grasp_id] < pos_thre:
                        print(" !!! arrive grasp position !!! ")
                        interp_pos = np.zeros(3)
                    
                    # interp_euler = Rotation.from_matrix(target_pose.to_transformation_matrix()[:3, :3] @ cur_pose.to_transformation_matrix()[:3, :3].T).as_euler('xyz') / 10
                    real_twist = np.concatenate((interp_pos, interp_euler)) * control_freq
                    
                    gripper_cmd = 0.085
                    print(f'==== step {step} twist, gripper_pos (base frame): {real_twist}, {gripper_cmd}')

                if isinstance(grasp_ee_final, np.ndarray):
                    real_twist[0] += self.final_velocity[0]
                    
                # robot execution
                self.robot.execute_combined_command(real_twist, gripper_cmd)
                print('command executed!')
                print("+"*10, time.time() - start)

        # stop
        print("Finish ! ! !")
        print("*** final velocity ****", self.final_velocity)
        arm_gripper_cmds = [
            [0.0, 0.0, 0.00, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.05, 0.05, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ]
        # self.robot.execute_combined_command(arm_gripper_cmds[0][:6], arm_gripper_cmds[0][6])
        # time.sleep(2)
        self.robot.execute_combined_command(arm_gripper_cmds[0][:6], arm_gripper_cmds[0][6])
        time.sleep(2)
        self.robot.execute_combined_command(arm_gripper_cmds[1][:6], arm_gripper_cmds[1][6])
        time.sleep(5)
        self.robot.execute_combined_command(arm_gripper_cmds[2][:6], arm_gripper_cmds[2][6])
        # self.vis_o3d.destroy_window()

    def test_rl(self, max_steps=100):
        trans_base2world = np.eye(4)
        trans_base2world[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        gripper_pts_ee = np.random.normal(0.0, 0.0425 / 3, size=(20, 3))
        control_freq = 5
        real_action = np.zeros(7)
        # self.scene_pc_ee, self.obj_pc_ee, self.grasp_ee, self.pred_grasps_ac, self.gripper_pts = [], [], [], [], []
        self.exp_info = defaultdict(list)

        with torch.no_grad():
            for step in tqdm(range(max_steps), colour='red', leave=False):
                self.step = step
                start = time.time()

                # get robot states
                robot_state_0 = self.robot.get_real_state()
                tcp_pose = robot_state_0['tcp_state']  # (pos, wxyz), base frame
                cur_ee_mat_world = trans_base2world @ Pose(p=tcp_pose[:3], q=tcp_pose[3:]).to_transformation_matrix()
                cur_ee_euler = Rotation.from_matrix(cur_ee_mat_world[:3, :3]).as_euler("XYZ")
                cur_obs = {
                    # :para gripper_pos: [0, 0.085], close -> open
                    'gripper_pos': np.repeat((1 - robot_state_0['gripper_pos'] / 0.085) * 0.0425, 2).astype(np.float32),  # [0, 0.0425]
                    'tcp_pose': np.hstack([cur_ee_mat_world[:3, 3], cur_ee_euler]).astype(np.float32),
                    # 'grasp_state': np.repeat(robot_state_0['grasp_state'], 5),
                    'action': real_action.astype(np.float32),
                }
                print("+++ gripper_pos: ", cur_obs['gripper_pos'])
                print("+++ tcp_pose: ", cur_obs['tcp_pose'])
                print("+++ action: ", cur_obs['action'])
                print("get real state time: ", time.time() - start)

                ## get grasps
                t = time.time()
                cur_pose_0 = Pose(p=robot_state_0['tcp_state'][:3], q=robot_state_0['tcp_state'][3:])
                trans_ee2base = cur_pose_0.to_transformation_matrix()
                # scene_rgbpc_base, obj_rgbpc_base, pred_gg_ee = self.gen_grasps(trans_ee2base=trans_ee2base)
                scene_points_ee, obj_points_ee = self.get_scene_points(trans_ee2base)
                print("+++ scene, obj points shape: ", scene_points_ee.shape, obj_points_ee.shape)
                print("get object points & filter time: ", time.time() - t)

                t = time.time()
                if obj_points_ee.shape[0] < 64:
                    pred_gg_ee = GraspGroup()
                else:
                    scene_pc_tensor, obj_pc_tensor = torch.from_numpy(scene_points_ee).cuda(), torch.from_numpy(obj_points_ee).cuda()
                    pred_gg_ee = self.lgNet.inference(
                        obj_points=obj_pc_tensor,
                        scene_points=scene_pc_tensor,
                        num_grasps=64,
                        scale=1.0
                    )
                    self.exp_info["pred_gg_ee"].append(pred_gg_ee)
                    pred_gg_ee.rotations = np.einsum('ijk, kl -> ijl', pred_gg_ee.rotations, np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]))  # (N, 4, 4)
                grasps_num = pred_gg_ee.size
                print('whole generate grasp time: ', time.time() - t)

                ## grasp sampling & transform to 3d points
                t = time.time()
                grasps_ee_0 = np.repeat(np.eye(4)[None], pred_gg_ee.size, axis=0)  # (N, 4, 4)
                grasps_ee_0[:, :3, 3] = pred_gg_ee.translations
                grasps_ee_0[:, :3, :3] = pred_gg_ee.rotations
                grasps_ee, grasps_scores = compute_near_grasps_rt(grasps_ee_0, pred_gg_ee.scores, num_grasps=self.num_grasps)
                R, T = grasps_ee[:, :3, :3].transpose((0, 2, 1)), grasps_ee[:, :3, 3]

                gripper_pts_diff = np.einsum("ij, kjl -> kil", gripper_pts_ee, R) + np.repeat(
                    T[:, None, :], gripper_pts_ee.shape[0], axis=1
                )  # (num, k, 3)
                grasp_exist = np.ones(5) if grasps_num > 0 else np.zeros(5)
                cur_obs.update(
                    {'grasp_exist': grasp_exist.astype(np.float32),
                     'gripper_pts_diff': gripper_pts_diff.astype(np.float32),
                     'grasps_scores': grasps_scores.astype(np.float32),
                     'close_grasp_pose_ee': np.zeros(7).astype(np.float32)}
                )
                print('grasp sampling & transform to 3d points, time', time.time() - t)

                ## RL prediction & grasp goal prediction
                t = time.time()
                action, _states = self.rl_model.predict(cur_obs, deterministic=True)
                real_action = clip_and_scale_action(action, self.action_bounds[0], self.action_bounds[1])
                obs_tensor = {k: torch.tensor(o[None], device="cuda:0") for k, o in cur_obs.items()}
                action_tensor = torch.tensor(action[None], device="cuda:0")
                _, pred_pose_actor = self.rl_model.actor.action_log_prob(obs_tensor)
                _, pred_pose_critic = self.rl_model.critic(obs_tensor, action_tensor)
                pred_grasp_actor_critic = torch.cat((pred_pose_actor, pred_pose_critic), dim=0).cpu().numpy()  # (2, 7)
                pred_grasp_ac_rotmat = Rotation.from_quat(np.roll(pred_grasp_actor_critic[:, :4], -1)).as_matrix()
                pred_grasp_ac = GraspGroup(
                    translations=pred_grasp_actor_critic[:, 4:],
                    rotations=pred_grasp_ac_rotmat @ np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]),
                    widths=np.array([0.085, 0.085]),
                    depths=np.array([0.02, 0.02]),
                    scores=np.array([1, 0])
                )
                self.exp_info["pred_grasps_ac"].append(pred_grasp_ac)
                print('RL & grasp goal prediction: ', time.time() - t)

                ## get real control command (action => twist in base frame)
                t = time.time()
                delta_pos, delta_euler, gripper_action = real_action[0:3], real_action[3:6], real_action[6]
                delta_quat = np.roll(Rotation.from_euler('XYZ', delta_euler).as_quat(), 1)
                print("delta pos, delta euler (align EE frame): ", delta_pos, delta_euler)
                target_pose_0 = cur_pose_0 * sapien.Pose(p=delta_pos, q=delta_quat)

                twist_base = target_pose_0 * cur_pose_0.inv()
                vel, wvel = target_pose_0.p - cur_pose_0.p, euler_from_quaternion(np.roll(twist_base.q, -1))
                twist = np.concatenate((vel, wvel))
                real_twist = twist * control_freq
                # real_twist[3:] = 0
                gripper_cmd = 0.085 - gripper_action
                print('action -> base twist, time: ', time.time() - t)

                ## robot execution
                self.robot.execute_combined_command(real_twist, gripper_cmd)
                print('command executed!')
                print("+"*10, time.time() - start)

        # stop
        print("Finish ! ! !")
        arm_gripper_cmds = [
            [0.0, 0.0, 0.00, 0.0, 0.0, 0.0, 0.085],
            [0.0, 0.05, 0.05, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ]
        self.robot.execute_combined_command(arm_gripper_cmds[0][:6], arm_gripper_cmds[0][6])
        time.sleep(2)
        # self.robot.execute_combined_command(arm_gripper_cmds[1][:6], arm_gripper_cmds[1][6])
        # time.sleep(5)
        # self.robot.execute_combined_command(arm_gripper_cmds[2][:6], arm_gripper_cmds[2][6])

        info_len = len(self.exp_info["scene_pc_ee"])
        # for i in range(info_len - 5, info_len - 1, 1):
        for i in range(info_len):
            scene_pc_ee = self.exp_info["scene_pc_ee"][i]
            obj_pc_ee = self.exp_info["obj_pc_ee"][i]
            pred_grasp_ac = self.exp_info["pred_grasps_ac"][i]
            pred_gg_ee = self.exp_info["pred_gg_ee"][i]
            draw_o3d_geometries(
                [np.concatenate((obj_pc_ee + np.array([0, 0, -0.001]), np.array([[1, 0, 0]]).repeat(obj_pc_ee.shape[0], 0)), axis=1),
                 np.concatenate((scene_pc_ee, np.array([[0, 0, 1]]).repeat(scene_pc_ee.shape[0], 0)), axis=1)] +
                pred_gg_ee.to_open3d_geometry_list() +
                pred_grasp_ac.to_open3d_geometry_list(size=4)
            )
        # self.vis_o3d.destroy_window()

    def test_gaddpg(self, max_steps=100):
        trans_base2world = np.eye(4)
        trans_base2world[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        control_freq = 5
        real_action = np.zeros(7)
        self.exp_info = defaultdict(list)

        prev_grasp_th = 0.05
        min_grasp_tcp_dist = np.inf

        ## move to prev_target_grasp
        with torch.no_grad():
            for step in tqdm(range(max_steps), colour='red', leave=False):
                self.step = step
                start = time.time()

                if min_grasp_tcp_dist > prev_grasp_th:
                    # get robot states
                    robot_state_0 = self.robot.get_real_state()
                    tcp_pose = robot_state_0['tcp_state']  # (pos, wxyz), base frame
                    cur_ee_mat_world = trans_base2world @ Pose(p=tcp_pose[:3], q=tcp_pose[3:]).to_transformation_matrix()
                    cur_ee_euler = Rotation.from_matrix(cur_ee_mat_world[:3, :3]).as_euler("XYZ")
                    cur_obs = {
                        # :para gripper_pos: [0, 0.085], close -> open
                        'gripper_pos': np.repeat((1 - robot_state_0['gripper_pos'] / 0.085) * 0.0425, 2).astype(np.float32),  # [0, 0.0425]
                        'tcp_pose': np.hstack([cur_ee_mat_world[:3, 3], cur_ee_euler]).astype(np.float32),
                        'action': real_action.astype(np.float32),
                    }
                    print("+++ gripper_pos: ", cur_obs['gripper_pos'])
                    print("+++ tcp_pose: ", cur_obs['tcp_pose'])
                    print("+++ action: ", cur_obs['action'])
                    print("get real state time: ", time.time() - start)

                    # get obj pc
                    t = time.time()
                    cur_pose_0 = Pose(p=robot_state_0['tcp_state'][:3], q=robot_state_0['tcp_state'][3:])
                    trans_ee2base = np.eye(4)
                    trans_ee2base[:3, 3] = robot_state_0['tcp_state'][:3]
                    trans_ee2base[:3, :3] = Rotation.from_quat(np.roll(robot_state_0['tcp_state'][3:], -1)).as_matrix()

                    scene_points_ee, obj_points_ee = self.get_scene_points(trans_ee2base)

                    print(obj_points_ee.shape)
                    pc_num = obj_points_ee.shape[0]
                    print("get object points & filter time: ", time.time() - t)

                    t = time.time()
                    objpc_num = 256
                    obj_pc_ee = np.zeros((objpc_num, 3))
                    if 0 <= pc_num <= objpc_num:
                        obj_pc_ee[:pc_num] = obj_points_ee
                    else:
                        random_ids = np.random.choice(
                            np.arange(pc_num), size=objpc_num, replace=False
                        )
                        obj_pc_ee = obj_points_ee[random_ids]

                    print('object points shape:', obj_points_ee.shape)
                    cur_obs.update(
                        {'obj_pc_ee': obj_pc_ee.astype(np.float32),
                         'close_grasp_pose_ee': np.zeros(7).astype(np.float32)}
                    )
                    print("update current obs time: ", time.time() - t)

                    # min_obj_tcp_dist = np.linalg.norm(obj_pc_ee, axis=1).min(axis=0)
                    # print("current min dist (obj <-> tcp): ", min_obj_tcp_dist)

                    ## RL & grasp goal prediction
                    t = time.time()
                    action, _states = self.rl_model.predict(cur_obs, deterministic=True)
                    real_action = clip_and_scale_action(action, self.action_bounds[0], self.action_bounds[1])  # [0, 0.0425], open -> close
                    self.exp_info["real_action"].append(np.append(real_twist, gripper_cmd))
                    obs_tensor = {k: torch.tensor(o[None], device="cuda:0") for k, o in cur_obs.items()}
                    action_tensor = torch.tensor(action[None], device="cuda:0")
                    _, pred_pose_actor = self.rl_model.actor.action_log_prob(obs_tensor)
                    _, pred_pose_critic = self.rl_model.critic(obs_tensor, action_tensor)
                    pred_grasp_actor_critic = torch.cat((pred_pose_actor, pred_pose_critic), dim=0).cpu().numpy()  # (2, 7)
                    pred_grasp_ac_rotmat = Rotation.from_quat(np.roll(pred_grasp_actor_critic[:, :4], -1)).as_matrix()  # (2, 3, 3)
                    pred_grasp_ac = GraspGroup(
                        translations=pred_grasp_actor_critic[:, 4:],
                        rotations=pred_grasp_ac_rotmat @ np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]),
                        widths=np.array([0.085, 0.085]),
                        depths=np.array([0.02, 0.02]),
                        scores=np.array([1, 0])
                    )
                    min_grasp_tcp_dist = np.linalg.norm(pred_grasp_actor_critic[0, 4:])
                    print("current min dist (pred_actor_grasp <-> tcp): ", min_grasp_tcp_dist)
                    self.exp_info["pred_grasps_ac"].append(pred_grasp_ac)
                    print("RL & grasp goal prediction time: ", time.time() - t)

                    ## save pn feat
                    self.exp_info["actor_pnfeat"].append(self.rl_model.actor.features_forward(obs_tensor)[0].cpu().numpy())
                    self.exp_info["critic_pnfeat"].append(self.rl_model.critic.features_forward(obs_tensor)[0].cpu().numpy())

                    ## action => twist (base frame)
                    t = time.time()
                    delta_pos, delta_euler, gripper_action = real_action[0:3], real_action[3:6], real_action[6]
                    delta_quat = np.roll(Rotation.from_euler('XYZ', delta_euler).as_quat(), 1)
                    print("delta pos, delta euler (align EE frame), gripper_action: ", delta_pos, delta_euler, gripper_action)
                    target_pose_0 = cur_pose_0 * sapien.Pose(p=delta_pos, q=delta_quat)

                    twist_base = target_pose_0 * cur_pose_0.inv()
                    vel, wvel = target_pose_0.p - cur_pose_0.p, euler_from_quaternion(np.roll(twist_base.q, -1))
                    twist = np.concatenate((vel, wvel))
                    real_twist = twist * control_freq
                    # real_twist[3:] = 0

                    gripper_cmd = 2 * (0.0425 - gripper_action)
                    # gripper_bin_thesh = 0.02
                    # gripper_cmd = 0 if gripper_action > gripper_bin_thesh else 0.085
                    # gripper_cmd = 0.085
                    print("twist (base frame), gripper_cmd: ", real_twist, gripper_cmd)
                    print("action => twist (base frame) time: ", time.time() - t)

                    # robot execution
                    self.robot.execute_combined_command(real_twist, gripper_cmd)  # gripper: [0, 0.085], close -> open
                    print('command executed!')
                    print("+"*10, time.time() - start)
                else:
                    # self.robot.execute_combined_command([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.085)
                    break

        ## move to target_grasp
        robot_state_0 = self.robot.get_real_state()
        cur_pose_0 = Pose(p=robot_state_0['tcp_state'][:3], q=robot_state_0['tcp_state'][3:])
        trans_ee2base = np.eye(4)
        trans_ee2base[:3, 3] = robot_state_0['tcp_state'][:3]
        trans_ee2base[:3, :3] = Rotation.from_quat(np.roll(robot_state_0['tcp_state'][3:], -1)).as_matrix()

        final_grasp_ee = Pose(p=pred_grasp_actor_critic[0, 4:], q=pred_grasp_actor_critic[0, :4])
        final_grasp_ee_mat = final_grasp_ee.to_transformation_matrix()
        final_grasp_ac = GraspGroup(
            translations=final_grasp_ee_mat[:3, 3][None],
            rotations=(final_grasp_ee_mat[:3, :3] @ np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]))[None],
            widths=np.array([0.085]),
            depths=np.array([0.02]),
            scores=np.array([1])
        )
        self.exp_info["pred_grasps_ac"].append(final_grasp_ac)

        final_grasp_base = Pose().from_transformation_matrix(trans_ee2base @ final_grasp_ee_mat)

        ## directly move
        # twist_base = final_grasp_base * cur_pose_0.inv()
        # vel, wvel = final_grasp_base.p - cur_pose_0.p, euler_from_quaternion(np.roll(twist_base.q, -1))
        # print(twist_base, vel, wvel)
        # twist = np.concatenate((vel, wvel))
        # real_twist = twist * control_freq
        # t = time.time()
        # self.robot.execute_combined_command(real_twist, 0.085)  # gripper: [0, 0.085], close -> open
        # exec_time = time.time() - t
        # print("actual control time: ", exec_time)
        # time.sleep(0.2 - exec_time)
        # print("open-loop control time: ", time.time() - t)
        ## move using interpolation
        interp_steps = 2
        interp_pos, interp_euler = pose_interp_steps(cur_pose_0, final_grasp_base, interp_steps=interp_steps)  # in base frame
        for i in range(interp_steps):
            t = time.time()
            real_twist = np.concatenate((interp_pos, interp_euler)) * control_freq
            self.robot.execute_combined_command(real_twist, 0.085)  # gripper: [0, 0.085], close -> open
            print("open-loop control time: ", time.time() - t)
        print('Move to final grasp pose!')

        ## close gripper & lift
        arm_gripper_cmds = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.03, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ] * control_freq
        # close gripper
        self.robot.execute_combined_command(arm_gripper_cmds[0][:6], arm_gripper_cmds[0][6])
        time.sleep(1)
        # lift the object
        self.robot.execute_combined_command(arm_gripper_cmds[1][:6], arm_gripper_cmds[1][6])
        time.sleep(5)
        # stop
        self.robot.execute_combined_command(arm_gripper_cmds[2][:6], arm_gripper_cmds[2][6])
        time.sleep(1)

        for i in range(self.step - 4, self.step - 1, 1):
            scene_pc_ee = self.exp_info["scene_pc_ee"][i]
            obj_pc_ee = self.exp_info["obj_pc_ee"][i]
            pred_grasp_ac = self.exp_info["pred_grasps_ac"][i]
            draw_o3d_geometries(
                [np.concatenate((obj_pc_ee + np.array([0, 0, -0.01]), np.array([[1, 0, 0]]).repeat(obj_pc_ee.shape[0], 0)), axis=1),
                 np.concatenate((scene_pc_ee, np.array([[0, 0, 1]]).repeat(scene_pc_ee.shape[0], 0)), axis=1)] +
                pred_grasp_ac.to_open3d_geometry_list(size=4)
            )

        scene_pc_ee = self.exp_info["scene_pc_ee"][-1]
        obj_pc_ee = self.exp_info["obj_pc_ee"][-1]
        final_grasp = self.exp_info["pred_grasps_ac"][-1]
        draw_o3d_geometries(
            [np.concatenate((obj_pc_ee + np.array([0, 0, -0.01]), np.array([[1, 0, 0]]).repeat(obj_pc_ee.shape[0], 0)), axis=1),
             np.concatenate((scene_pc_ee, np.array([[0, 0, 1]]).repeat(scene_pc_ee.shape[0], 0)), axis=1)] +
            final_grasp.to_open3d_geometry_list(size=4)
        )
        print("Finish ! ! !")
        # self.vis_o3d.destroy_window()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 'hand_sn': '138422075756', 'gripper_sn': '332522071141'
    parser.add_argument("--realsense-sn", type=str, nargs='+', default=['138422075756', '332522071141'])
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--ground-ws", type=json.loads, default=[[-0.4, 0.4], [-1.0, -0.3], [0.08 + 0.06, 0.5]])  # + 0.065 for C+T
    parser.add_argument("--object-z-offset", type=float, default=0.01)
    parser.add_argument("--rl-mode", type=str, default="grasprt")  # heuristic_plan, objpcrt, grasprt
    # ours: 20240324_161009, GA-DDPG: 20240309_220614
    parser.add_argument("--model-path", type=str, default=str(ALGORITHM_DIR / "scripts/20240324_161009"))

    ## heuristic planning
    parser.add_argument("--method", type=str, default="simple_interp")
    parser.add_argument("--task", type=str, default="dynamic")  # handover, dynamic

    ## ours
    parser.add_argument("--real-mode", type=str, default="ws_filter")  # explorer, ws_filter
    parser.add_argument("--ball-query-r", type=float, default=0.05)
    parser.add_argument("--trans-dist-th", type=float, default=0.07)
    parser.add_argument("--rot-dist-th", type=float, default=0.07)
    parser.add_argument("--speed-para", type=float, default=0.3)
    parser.add_argument("--num-centers", type=int, nargs='+', default=[64, 12])
    parser.add_argument("--traj-mode", type=str, default="CT_v004v6_obj2_demo_1")  # CT_v004v6_obj2_0, handover

    parser.add_argument("--save-rgb", type=bool, default=True)
    parser.add_argument("--vis-data", action='store_true')
    parser.add_argument("--visualizer", action='store_true')
    parser = lg_parse(parser)
    args = parser.parse_args()

    sim2real = Sim2Real(args=args)

    ## real test
    # robot_state = sim2real.robot.get_real_state()
    # print(robot_state)
    # cur_pose_0 = Pose(p=robot_state['tcp_state'][:3], q=robot_state['tcp_state'][3:])
    # trans_ee2base = cur_pose_0.to_transformation_matrix()
    # sim2real.test_cameras(trans_ee2base=trans_ee2base)
    # sim2real.get_scene_points(trans_ee2base=trans_ee2base)

    ## LoG warm start
    robot_state = sim2real.robot.get_real_state()
    print(robot_state)
    cur_pose_0 = Pose(p=robot_state['tcp_state'][:3], q=robot_state['tcp_state'][3:])
    trans_ee2base = cur_pose_0.to_transformation_matrix()
    if args.save_rgb:
        scene_points_base, obj_points_base, centers_base, scene_points_ee, obj_points_ee, centers_ee, pred_gg_ee, _, _ = \
        sim2real.init_grasps(
            trans_ee2base=trans_ee2base,
            vis=False
        )
    else:
        scene_points_base, obj_points_base, centers_base, scene_points_ee, obj_points_ee, centers_ee, pred_gg_ee = \
        sim2real.init_grasps(
            trans_ee2base=trans_ee2base,
            vis=False
        )
    print("Warm Start Down!")
    time.sleep(2)

    if args.rl_mode == "heuristic_plan":
        ## heuristic_plan
        sim2real.test_heuristic_plan()
    elif args.rl_mode == "grasprt":
        ## ours
        # sim2real.test_rl(
        #     max_steps=40,
        # )
        sim2real.test_ours()
    elif args.rl_mode == "objpcrt":
        ## gaddpg
        sim2real.test_gaddpg(
            max_steps=40,
        )
