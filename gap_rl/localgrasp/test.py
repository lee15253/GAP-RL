import argparse
import numpy as np
import torch
import open3d as o3d

from LoG import lg_parse, LgNet


if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=3)
    parser = argparse.ArgumentParser(add_help=False)
    parser = lg_parse(parser)
    args, opts = parser.parse_known_args()

    args.checkpoint_path = '/home/nas2_userI/byungkunlee/research/rl/GAP-RL/gap_rl/localgrasp/epoch_11_acc_0.915_cover_0.765'

    grasp_detector = LgNet(args)

    data = np.load('/home/nas2_userI/byungkunlee/research/rl/GAP-RL/gap_rl/localgrasp/scene_example.npz', allow_pickle=True)

    rgb, xyz, centers = data['arr_0'], data['arr_1'], data['arr_2']
    pred_gg = grasp_detector.infer_from_centers(
        scene_points=torch.from_numpy(xyz).float().cuda(),
        centers=torch.from_numpy(centers).float().cuda(),
    )
    centers_mesh = []
    for center in centers:
        mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
        mesh_sphere.paint_uniform_color([1, 0, 0])
        mesh_sphere.translate(center)
        centers_mesh.append(mesh_sphere)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd, frame] + centers_mesh + pred_gg.to_open3d_geometry_list())