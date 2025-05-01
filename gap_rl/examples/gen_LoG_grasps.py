import argparse
import os
from copy import deepcopy
from collections import OrderedDict, defaultdict
import gym
import mplib
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from tqdm import tqdm

import open3d as o3d
import matplotlib.pyplot as plt
import sapien.core as sapien
from sapien.core import Pose

from gap_rl.envs.base_env import BaseEnv
from gap_rl.envs.pick_single import PickSingleYCBEnv
from gap_rl.localgrasp.LoG import lg_parse, LgNet, GraspGroup
from gap_rl.utils.geometry import (
    angle_distance_ms,
    homo_transfer,
    sample_grasp_points_ee,
    sample_grasp_multipoints_ee,
    transform_points,
    pointcloud_filter,
)
from gap_rl.utils.io_utils import load_json, dump_json
from gap_rl.utils.o3d_utils import draw_o3d_geometries
from gap_rl.utils.sapien_utils import look_at
from gap_rl.utils.visualization.cv2_utils import visualize_depth


import faulthandler

faulthandler.enable()


def grasps_nms(grasp_file, model_ids, nms=False, vis=True, save=False):
    model_grasps = load_json(grasp_file)
    env: BaseEnv = gym.make(
        "PickSingleYCB-v0",
        shader_dir="ibl",
        robot="ur5e_robotiq85_old",
        model_ids=["003_cracker_box"],
        obj_init_rot_z=True,
        obs_mode="state_egopoints",
        reward_mode="dense",
        control_mode="pd_ee_delta_pose",
        sim_freq=150,
        control_freq=5,
        gen_traj_mode="line",  # None, "line", "circle", "random", "random3d"
        vary_speed=True,
        robot_x_offset=0.56,
    )

    for model_id in model_ids:
        _ = env.reset(model_id=model_id)
        # action_sample = env.action_space.sample()
        # action = np.zeros(action_sample.shape)
        # # action = np.zeros(7)
        # obs, rew, done, info = env.step(action)
        env.update_render()
        obj_grasp = model_grasps[model_id]
        grasps_num = len(obj_grasp["widths"])
        print(f"{model_id}，grasps num: ", grasps_num)
        trans_obj2world = env.obj_pose.to_transformation_matrix()
        grasps_mat_obj = np.array(obj_grasp["transformations"])
        obj_grasp = GraspGroup(
            translations=grasps_mat_obj[:, :3, 3],
            rotations=grasps_mat_obj[:, :3, :3],
            heights=np.array(obj_grasp["heights"]),
            widths=np.array(obj_grasp["widths"]),
            depths=np.array(obj_grasp["depths"]),
            scores=np.array(obj_grasp["scores"]),
        )
        if nms:
            obj_grasp_gg = obj_grasp.to_graspnet_gg()
            print("before nms: ", len(obj_grasp_gg))
            obj_grasp_gg = obj_grasp_gg.nms(0.01, 60 / 180 * np.pi)
            print("after nms: ", len(obj_grasp_gg))
            obj_grasp = GraspGroup(
                translations=obj_grasp_gg.translations,
                rotations=obj_grasp_gg.rotation_matrices,
                heights=obj_grasp_gg.heights,
                widths=obj_grasp_gg.widths,
                depths=obj_grasp_gg.depths,
                scores=obj_grasp_gg.scores,
            )
        obj_grasp = obj_grasp.to_list()
        model_grasps[model_id] = obj_grasp

        if vis:
            trans_world2ee = env.tcp.pose.inv().to_transformation_matrix()
            trans_obj2ee = trans_world2ee @ trans_obj2world  # obj -> ee
            # grasps_mat_obj = obj_grasp["transformations"]
            draw_gg_ee = np.einsum("ij, kjl -> kil", trans_obj2ee, grasps_mat_obj)
            draw_gg_ee[..., :3, :3] = np.einsum(
                "ijk, kl->ijl", draw_gg_ee[..., :3, :3], np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
            )
            draw_grasp = GraspGroup(
                translations=draw_gg_ee[:, :3, 3],
                rotations=draw_gg_ee[:, :3, :3],
                heights=np.array(obj_grasp["heights"]),
                widths=np.array(obj_grasp["widths"]),
                depths=np.array(obj_grasp["depths"]),
                scores=np.array(obj_grasp["scores"]),
            )
            # draw_grasp.scores[0] = 1  # set the best grasp red
            grasps_geo_ee = draw_grasp.to_open3d_geometry_list(size=2)[:20]
            obj_pc_ee = transform_points(trans_obj2ee, env.obj_pc)
            vispc_ee = o3d.geometry.PointCloud()
            vispc_ee.points = o3d.utility.Vector3dVector(obj_pc_ee)
            vispc_ee.paint_uniform_color([0, 0, 1])
            draw_o3d_geometries([vispc_ee] + grasps_geo_ee)

    if save:
        dump_json(grasp_file, model_grasps)


def main(grasp_file, model_ids, stereo=False, vis=False, render=False, save=False):
    np.set_printoptions(suppress=True, precision=3)
    gripper_pos = [0, 0.02125, 0.0425]

    env: BaseEnv = gym.make(
        "PickSingleYCB-v1",
        shader_dir="ibl",
        robot="ur5e_robotiq85_old",
        model_ids=model_ids,
        obj_init_rot_z=True,
        obs_mode="state_objpoints_rt",
        reward_mode="dense",
        control_mode="pd_ee_delta_pose",
        sim_freq=150,
        control_freq=5,
        gen_traj_mode="random2d",  # None, "line", "circle", "random", "random3d"
        vary_speed=True,
        robot_x_offset=0.56,
    )
    # env.set_stereo_mode(is_stereo=stereo)  # generate more realistic depth / point clouds

    print("Action space", env.action_space)
    print("Observation mode", env.obs_mode)
    print("Control mode", env.control_mode)
    print("Reward mode", env.reward_mode)

    # setup LoG
    # lg_args = lg_parse_args()
    parser = argparse.ArgumentParser()
    parser = lg_parse(parser)
    lg_args = parser.parse_args()
    # lg_args.vis = vis
    lgNet = LgNet(lg_args)

    # from hand_realsense
    # trans_cam2ee = np.array(
    #     [
    #         [1, 0, 0, -0.032],
    #         [0, -1, 0, -0.064],
    #         [0, 0, -1, -0.15],
    #         [0, 0, 0, 1],
    #     ]
    # )
    # trans_cam2ee = np.array(
    #     [
    #         [1, 0, 0, -0.0251],
    #         [0, -1, 0, -0.1178],
    #         [0, 0, -1, -0.19],
    #         [0, 0, 0, 1],
    #     ]
    # )

    model_grasps = OrderedDict()
    pred_grasps = OrderedDict()

    rot_xy_range = np.linspace(10 / 180 * np.pi, 80 / 180 * np.pi, 6)
    rot_z_range = np.linspace(-np.pi, np.pi, 13)[:12]  # 60
    grid_sample = np.array(np.meshgrid(rot_xy_range, rot_z_range)).reshape(2, -1)
    model_grasps["rot_grid"] = grid_sample  # (2, )
    if save:
        viser = o3d.visualization.Visualizer()
        viser.create_window(width=1280, height=720)
        vis_dir = "ycb_train"
        os.makedirs(vis_dir, exist_ok=True)

    for model_id in model_ids:
        model_grasps[model_id] = OrderedDict()
        pred_grasps[model_id] = OrderedDict()
        if save:
            img_dir = f"./{vis_dir}/{model_id}"
            os.makedirs(img_dir, exist_ok=True)

        print(f"============ {model_id} ============")
        _ = env.reset(model_id=model_id)
        obj_center_pos = env.obj_init_pos

        if render:
            viewer = env.render(mode="human")
            print("Press [s] to start")
            while True:
                if viewer.window.key_down("s"):
                    break
                env.render()
        else:
            env.update_render()

        length, width, height = env.obj_aabb_halfsize
        dist = np.sqrt(width**2 + length**2) / 2 + 0.5
        height_noise = height / 2 + np.random.normal(0, height / 30)
        # print(np.abs(height_noise - height / 2))
        sample_pos = np.stack(
            (
                -dist * np.cos(grid_sample[0]) * np.cos(grid_sample[1]),
                dist * np.cos(grid_sample[0]) * np.sin(grid_sample[1]),
                dist * np.sin(grid_sample[0]) + height_noise,
            ),
            axis=1,
        )
        sample_campos = sample_pos[sample_pos[:, 2] > 0.01]

        obj_grasps_list = []
        pred_grasps_list = []
        for ind, pos_it in enumerate(sample_campos):
            print(f"--------- {pos_it}, {grid_sample[:, ind]} ---------")
            sample_poscenter = obj_center_pos + np.hstack(
                (
                    np.random.normal(loc=0, scale=0.01, size=1),
                    np.random.normal(loc=0, scale=0.01, size=1),
                    np.random.normal(loc=0, scale=0.01, size=1),
                )
            )

            # random sample camera in-plane rotation
            # sample_cam_roty = (np.random.rand() - 0.5) * 2  # (-1, 1)
            # sample_cam_rotz = np.sqrt(1 - sample_cam_roty**2) * np.random.choice([-1, 1])
            sample_cam_roty, sample_cam_rotz = 0, 1
            data_cam_pose = look_at(eye=pos_it, target=sample_poscenter, up=[0, sample_cam_roty, sample_cam_rotz])
            # cam_rot_mat = data_camlink_pose.to_transformation_matrix()[:3,:3] @ np.array([[0,0,1],[-1,0,0],[0,-1,0]])
            # data_cam_pose = sapien.Pose().from_transformation_matrix(homo_transfer(R=cam_rot_mat, T=data_camlink_pose.p))
            env.unwrapped._cameras["data_cam"].camera.set_pose(data_cam_pose)

            if render:
                env.render()

            # cam = env.unwrapped._cameras["data_cam"]
            # rgb, depth = cam.camera.get_rgb(), cam.camera.get_depth()
            # plt.subplot(1, 2, 1)
            # plt.imshow(rgb)
            # plt.subplot(1, 2, 2)
            # plt.imshow(depth)
            # plt.show()

            obs = env.get_state_objpoints_rt(action=None)
            obj_pc_ee, scene_pc_ee = obs["obj_pc_ee"], obs["scene_pc_ee"]
            if vis:
                draw_o3d_geometries([scene_pc_ee])
                draw_o3d_geometries([obj_pc_ee])

            print("obj points shape: ", obj_pc_ee.shape[0])
            if obj_pc_ee.shape[0] < 64:
                print("too few object points")
                obj_grasps_list.append(None)
                pred_grasps_list.append(None)
                continue
            pred_gg = lgNet.inference(
                obj_points=torch.from_numpy(obj_pc_ee).to("cuda:0"),
                scene_points=torch.from_numpy(scene_pc_ee.copy()).to("cuda:0"),
                num_grasps=64,
                scale=1.0,
            )
            print("LoG grasps number:", pred_gg.size)

            # get grasp from model
            if pred_gg.size == 0:
                print("No grasps generated!")
                obj_grasps_list.append(None)
                pred_grasps_list.append(None)
                continue

            # ee to world
            if env.unwrapped.use_stereo:
                trans_cam2world = env.unwrapped._cameras["data_cam"].camera._cam_rgb.get_model_matrix()
            else:
                trans_cam2world = env.unwrapped._cameras["data_cam"].camera.get_model_matrix()
            trans_world2obj = env.obj_pose.inv().to_transformation_matrix()
            # ee -> cam (맨오른쪽) / cam -> world  (그왼쪽) / world -> obj (그왼쪽)
            trans_ee2obj = trans_world2obj @ trans_cam2world @ np.linalg.inv(env.unwrapped.trans_cam2ee)
            # trans_ee2world = trans_cam2world @ np.linalg.inv(env.unwrapped.trans_cam2ee)

            T = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
            rot_mat = pred_gg.rotations
            pred_gg.rotations = np.einsum("ijk, kl->ijl", rot_mat, T)
            pred_grasps_list.append(pred_gg.to_list())

            gg_homo = homo_transfer(R=pred_gg.rotations, T=pred_gg.translations)
            # X_gg_homo = eeToobj @ ee_gg_homo
            X_gg_homo = np.einsum("ij, kjl->kil", trans_ee2obj, gg_homo)
            trans_gg = deepcopy(pred_gg)
            trans_gg.translations = X_gg_homo[:, :3, 3]
            trans_gg.rotations = X_gg_homo[:, :3, :3]

            obj_grasp = GraspGroup(
                translations=trans_gg.translations,
                rotations=trans_gg.rotations,
                heights=trans_gg.heights,
                widths=trans_gg.widths,
                depths=trans_gg.depths,
                scores=trans_gg.scores,
            )
            obj_grasps_list.append(obj_grasp.to_list())

            if vis or save:
                draw_gg = deepcopy(pred_gg)
                rot_mat = draw_gg.rotations
                draw_gg.rotations = np.einsum("ijk, kl->ijl", rot_mat, T.T)
                drawgg_homo = homo_transfer(R=draw_gg.rotations, T=draw_gg.translations)
                X_drawgg_homo = np.einsum("ij, kjl -> kil", trans_ee2obj, drawgg_homo)
                draw_gg.translations = X_drawgg_homo[:, :3, 3]
                draw_gg.rotations = X_drawgg_homo[:, :3, :3]

                draw_gg.scores[0] = 1  # set the best grasp red
                grasps_geo = draw_gg.to_open3d_geometry_list(size=1)
                vispc = o3d.geometry.PointCloud()
                vispc.points = o3d.utility.Vector3dVector(transform_points(trans_ee2obj, scene_pc_ee))
                vispc.paint_uniform_color([0, 0, 1])
                vispc_obj = o3d.geometry.PointCloud()
                vispc_obj.points = o3d.utility.Vector3dVector(transform_points(trans_ee2obj, obj_pc_ee))
                vispc_obj.paint_uniform_color([0, 1, 1])
                gripper_pts_ee = sample_grasp_points_ee([gripper_pos[2], gripper_pos[2]], z_offset=0.03)
                gripper_pts_obj = transform_points(trans_ee2obj, gripper_pts_ee)
                o3d_spheres = []
                for pts in gripper_pts_obj:
                    gripper = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
                    gripper.translate(pts)
                    gripper.paint_uniform_color([1, 0, 0])
                    o3d_spheres.append(gripper)
                XYZ = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
                if vis:
                    o3d.visualization.draw_geometries([XYZ, vispc, vispc_obj] + o3d_spheres + grasps_geo)
                if save:
                    imgs = env.unwrapped._cameras["data_cam"].get_images()
                    plt.figure()
                    plt.subplot(1, 2, 1)
                    plt.imshow(imgs["Color"])
                    plt.title("rgb")
                    plt.axis("off")
                    plt.subplot(1, 2, 2)
                    plt.imshow(visualize_depth(imgs["Position"].squeeze()))
                    plt.title("depth")
                    plt.axis("off")
                    if save:
                        plt.savefig(f"{img_dir}/{ind}_{model_id}.jpg", bbox_inches="tight", pad_inches=0.0)
                    # else:
                    #     plt.show()
                    plt.close()

                    viser.clear_geometries()
                    for geo in grasps_geo:
                        viser.add_geometry(geo)
                    for pts in gripper_pts_obj:
                        gripper = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
                        gripper.translate(pts)
                        gripper.paint_uniform_color([1, 0, 0])
                        viser.add_geometry(gripper)
                    viser.add_geometry(vispc)
                    viser.add_geometry(vispc_obj)
                    viser.add_geometry(XYZ)
                    ctr = viser.get_view_control()
                    ctr.set_front((1, 0, 2))
                    ctr.set_up((0, 1, 0))
                    ctr.set_lookat((0, 0, 0))
                    viser.poll_events()
                    viser.update_renderer()
                    filename = os.path.join(img_dir, f"{ind}_{model_id}_grasp_{len(draw_gg)}.png")
                    viser.capture_screen_image(filename, do_render=True)

        if save:
            model_grasps[model_id]["grasp"] = obj_grasps_list
            pred_grasps[model_id]["grasp"] = obj_grasps_list
    if save:
        dump_json(vis_dir + grasp_file, model_grasps)
        dump_json(vis_dir + "/preg_grasps_ee.json", pred_grasps)

    if save or vis:
        viser.destroy_window()


if __name__ == "__main__":
    # fmt: off
    model_ids = [
        "003_cracker_box", "005_tomato_soup_can", "010_potted_meat_can", "057_racquetball",
        "006_mustard_bottle", "024_bowl", "025_mug", "072-b_toy_airplane",
        "011_banana", "014_lemon", "043_phillips_screwdriver", "072-d_toy_airplane",
    ]
    # fmt: on
    main(
        grasp_file="/info_localgrasp_v3.json",  # info_localgrasp_v1, info_graspnet_v0
        model_ids=model_ids,
        stereo=False,
        vis=False,
        render=False,
        save=True,
    )
    # grasps_nms(
    #     grasp_file='info_localgrasp_v1.json',  # info_localgrasp_v1, info_graspnet_v0
    #     model_ids=model_ids,
    #     nms=True,
    #     vis=False,
    #     save=True
    # )
