from typing import Dict, List

import numpy as np
from copy import deepcopy
import sapien.core as sapien
import torch

from gap_rl.utils.bounding_cylinder import aabc
from sapien.core import Actor, Articulation, Link, Pose
from scipy.spatial.transform import Rotation


def sample_on_unit_sphere(rng):
    """
    Algo from http://corysimon.github.io/articles/uniformdistn-on-sphere/
    """
    v = np.zeros(3)
    while np.linalg.norm(v) < 1e-4:
        v[0] = rng.normal()  # random standard normal
        v[1] = rng.normal()
        v[2] = rng.normal()

    v = v / np.linalg.norm(v)
    return v


def sample_on_unit_circle(rng):
    v = np.zeros(2)
    while np.linalg.norm(v) < 1e-4:
        v[0] = rng.normal()  # random standard normal
        v[1] = rng.normal()

    v = v / np.linalg.norm(v)
    return v


def sample_grasp_points_ee(gripper_pos, z_offset=0.03):
    """
    sample 6 points representing gripper, in EE frame
    """
    x0_trans, x1_trans = gripper_pos[0] + 0.002, gripper_pos[1] + 0.002
    finger_points = np.array([
        [0, 0, -0.14],
        [0, 0, -0.07],
        [x0_trans, 0, -0.07],
        [x0_trans, 0, 0],
        [-x1_trans, 0, -0.07],
        [-x1_trans, 0, 0],
    ])
    adjust_finger_points = finger_points + np.array([0, 0, z_offset])
    return adjust_finger_points


def sample_grasp_multipoints_ee(gripper_pos, num_points_perlink=10, z_offset=0.03):
    """
    sample 6 points representing gripper, in EE frame
    """
    x_trans = gripper_pos + 0.002
    left_finger = np.linspace([-x_trans, 0, 0], [-x_trans, 0, -0.07], num_points_perlink)[:-1]
    left_knuckle = np.linspace([-x_trans, 0, -0.07], [0, 0, -0.07], num_points_perlink)[:-1]
    right_finger = np.linspace([x_trans, 0, 0], [x_trans, 0, -0.07], num_points_perlink)[:-1]
    right_knuckle = np.linspace([x_trans, 0, -0.07], [0, 0, -0.07], num_points_perlink)[:-1]
    base = np.linspace([0, 0, -0.14], [0, 0, -0.07], num_points_perlink)
    finger_points = np.concatenate((
        base, left_knuckle, left_finger, right_knuckle, right_finger
    ))
    adjust_finger_points = finger_points + np.array([0, 0, z_offset])
    return adjust_finger_points


def sample_grasp_keypoints_ee(gripper_w, num_points_perlink=2):
    """
    sample 4k points representing gripper, in EE frame
    """
    left_lib = np.linspace([-gripper_w, 0, gripper_w/2], [-gripper_w, 0, -gripper_w/2], num_points_perlink + 1)[:-1]
    top_lib = np.linspace([-gripper_w, 0, -gripper_w/2], [gripper_w, 0, -gripper_w/2], num_points_perlink + 1)[:-1]
    right_lib = np.linspace([gripper_w, 0, -gripper_w/2], [gripper_w, 0, gripper_w/2], num_points_perlink + 1)[:-1]
    bottom_lib = np.linspace([gripper_w, 0, gripper_w/2], [-gripper_w, 0, gripper_w/2], num_points_perlink + 1)[:-1]
    finger_points = np.concatenate((
        left_lib, top_lib, right_lib, bottom_lib
    ))
    return finger_points


def sample_query_grasp_points_ee(rng, gripper_pos, num_points=6):
    """
    sample 6 points representing gripper, in EE frame
    """
    scale = gripper_pos / 3
    # query_grasp_points = np.random.normal(0.0, scale, size=(num_points, 3))
    query_grasp_points = rng.normal(0.0, scale, size=(num_points, 3))
    return query_grasp_points


def rotation_between_vec(a, b):  # from a to b
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    axis = np.cross(a, b)
    axis = axis / np.linalg.norm(axis)  # norm might be 0
    angle = np.arccos(a @ b)
    R = Rotation.from_rotvec(axis * angle)
    return R


def angle_between_vec(a, b):  # from a to b
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    angle = np.arccos(a @ b)
    return angle


def wxyz_to_xyzw(q):
    return np.concatenate([q[1:4], q[0:1]])


def xyzw_to_wxyz(q):
    return np.concatenate([q[3:4], q[0:3]])


def qmul(q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
    """multiply two quaternion.

    Args:
        q0 (np.array): (1, 4)
        q1 (np.array): (1, 4)

    Returns:
        q (np.array): (1, 4)
    """
    w0, x0, y0, z0 = q0
    w1, x1, y1, z1 = q1
    return np.array([
        -x0 * x1 - y0 * y1 - z0 * z1 + w0 * w1, x0 * w1 + y0 * z1 - z0 * y1 + w0 * x1,
        -x0 * z1 + y0 * w1 + z0 * x1 + w0 * y1, x0 * y1 - y0 * x1 + z0 * w1 + w0 * z1
    ])


def rotate_2d_vec_by_angle(vec, theta):
    rot_mat = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    return rot_mat @ vec


def angle_distance(q0: sapien.Pose, q1: sapien.Pose):
    qd = (q0.inv() * q1).q
    return 2 * np.arctan2(np.linalg.norm(qd[1:]), qd[0]) / np.pi


def angle_distance_ms(q0: np.ndarray, q1: np.ndarray):
    assert q0.size == 4 and q1.size == 4, f"quat item is of length 4"
    theta = 2 * np.arccos(np.clip(np.abs(q0 @ q1), a_min=0, a_max=1))
    return theta


def angle_distance_simple(q0: np.ndarray, q1: np.ndarray):
    assert q0.size == 4 and q1.size == 4, f"quat item is of length 4"
    return 1 - np.clip(np.abs(q0 @ q1), a_min=0, a_max=1)


def get_axis_aligned_bbox_for_articulation(art: Articulation):
    mins = np.array([np.inf, np.inf, np.inf])
    maxs = -mins
    for link in art.get_links():
        lp = link.pose
        for s in link.get_collision_shapes():
            p = lp * s.get_local_pose()
            T = p.to_transformation_matrix()
            vertices = s.geometry.vertices * s.geometry.scale
            vertices = vertices @ T[:3, :3].T + T[:3, 3]
            mins = np.minimum(mins, vertices.min(0))
            maxs = np.maximum(maxs, vertices.max(0))
    return mins, maxs


def get_axis_aligned_bbox_for_actor(actor: Actor):
    mins = np.ones(3) * np.inf
    maxs = -mins

    for shape in actor.get_collision_shapes():  # this is CollisionShape
        scaled_vertices = shape.geometry.vertices * shape.geometry.scale
        local_pose = shape.get_local_pose()
        mat = (actor.get_pose() * local_pose).to_transformation_matrix()
        world_vertices = scaled_vertices @ (mat[:3, :3].T) + mat[:3, 3]
        mins = np.minimum(mins, world_vertices.min(0))
        maxs = np.maximum(maxs, world_vertices.max(0))

    return mins, maxs


def get_local_axis_aligned_bbox_for_link(link: Link):
    mins = np.array([np.inf, np.inf, np.inf])
    maxs = -mins
    for s in link.get_collision_shapes():
        p = s.get_local_pose()
        T = p.to_transformation_matrix()
        vertices = s.geometry.vertices * s.geometry.scale
        vertices = vertices @ T[:3, :3].T + T[:3, 3]
        mins = np.minimum(mins, vertices.min(0))
        maxs = np.maximum(maxs, vertices.max(0))
    return mins, maxs


def get_local_aabc_for_actor(actor):
    all_vertices = []
    for s in actor.get_collision_shapes():
        p = s.get_local_pose()
        T = p.to_transformation_matrix()
        vertices = s.geometry.vertices * s.geometry.scale
        vertices = vertices @ T[:3, :3].T + T[:3, 3]
        all_vertices.append(vertices)
    vertices = np.vstack(all_vertices)
    return aabc(vertices)


def transform_points(H, pts):
    assert H.shape == (4, 4), H.shape
    assert pts.ndim == 2 and pts.shape[1] >= 3, pts.shape
    if pts.shape[1] > 3:
        # other state extended
        pts_pos = pts[:, :3]
        trans_pos = pts_pos @ H[:3, :3].T + H[:3, 3]
        if isinstance(pts, np.ndarray):
            trans_pts = np.concatenate((trans_pos, pts[:, 3:]), axis=1)
        elif isinstance(pts, torch.Tensor):
            trans_pts = torch.concat((trans_pos, pts[:, 3:]), dim=1)
        else:
            raise NotImplementedError
    else:
        trans_pts = pts @ H[:3, :3].T + H[:3, 3]
    return trans_pts


def invert_transform(H: np.ndarray):
    assert H.shape[-2:] == (4, 4), H.shape
    H_inv = H.copy()
    R_T = np.swapaxes(H[..., :3, :3], -1, -2)
    H_inv[..., :3, :3] = R_T
    H_inv[..., :3, 3:] = -R_T @ H[..., :3, 3:]
    return H_inv


def homo_transfer(R: np.ndarray, T: np.ndarray):
    """
    R, T shape: [N, 3, 3], [N, 3]
    or R, T shape: [3, 3], [3]
    """
    if len(R.shape) == 3:
        assert R.shape[0] == T.shape[0] and R.shape[1:] == (3, 3) and T.shape[1:] == (3,)
        H = np.zeros((R.shape[0], 4, 4))
        H[:, :3, :3] = R
        H[:, :3, 3] = T
        H[:, 3, 3] = 1
    elif len(R.shape) == 2:
        assert R.shape == (3, 3) and T.shape == (3,)
        H = np.eye(4)
        H[:3, :3] = R
        H[:3, 3] = T
    return H


def pose_to_posrotvec(pose: sapien.Pose):
    pos, quat = pose.p, pose.q
    rotvec = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_rotvec()
    return np.append(pos, rotvec)


def mat_to_posrotvec(mat: np.ndarray):
    assert mat.shape[-2:] == (4, 4)
    pos, rotmat = mat[..., :3, 3], mat[..., :3, :3]
    rotvec = Rotation.from_matrix(rotmat).as_rotvec()
    return np.concatenate((pos, rotvec), axis=-1)


def get_oriented_bounding_box_for_2d_points(
    points_2d: np.ndarray, resolution=0.0
) -> Dict:
    assert len(points_2d.shape) == 2 and points_2d.shape[1] == 2
    if resolution > 0.0:
        points_2d = np.round(points_2d / resolution) * resolution
        points_2d = np.unique(points_2d, axis=0)
    ca = np.cov(points_2d, y=None, rowvar=0, bias=1)

    v, vect = np.linalg.eig(ca)
    tvect = np.transpose(vect)

    # use the inverse of the eigenvectors as a rotation matrix and
    # rotate the points so they align with the x and y axes
    ar = np.dot(points_2d, np.linalg.inv(tvect))

    # get the minimum and maximum x and y
    mina = np.min(ar, axis=0)
    maxa = np.max(ar, axis=0)
    half_size = (maxa - mina) * 0.5

    # the center is just half way between the min and max xy
    center = mina + half_size
    # get the 4 corners by subtracting and adding half the bounding boxes height and width to the center
    corners = np.array(
        [
            center + [-half_size[0], -half_size[1]],
            center + [half_size[0], -half_size[1]],
            center + [half_size[0], half_size[1]],
            center + [-half_size[0], half_size[1]],
        ]
    )

    # use the the eigenvectors as a rotation matrix and
    # rotate the corners and the centerback
    corners = np.dot(corners, tvect)
    center = np.dot(center, tvect)

    return {"center": center, "half_size": half_size, "axes": vect, "corners": corners}


def pointcloud_filter(points, *xyz_min_max):
    """
    :para points: [N, 3 + K]
    :para xyz_min_max: [3, 2], min and max of x, y, z
    """
    xyz_min_max = xyz_min_max[0]
    if isinstance(points, np.ndarray):
        mask1 = np.logical_and(points[:, 0] > xyz_min_max[0][0], points[:, 0] < xyz_min_max[0][1])
        mask2 = np.logical_and(points[:, 1] > xyz_min_max[1][0], points[:, 1] < xyz_min_max[1][1])
        mask3 = np.logical_and(points[:, 2] > xyz_min_max[2][0], points[:, 2] < xyz_min_max[2][1])
        mask = np.logical_and(np.logical_and(mask1, mask2), mask3)
    elif isinstance(points, torch.Tensor):
        mask1 = torch.logical_and(points[:, 0] > xyz_min_max[0][0], points[:, 0] < xyz_min_max[0][1])
        mask2 = torch.logical_and(points[:, 1] > xyz_min_max[1][0], points[:, 1] < xyz_min_max[1][1])
        mask3 = torch.logical_and(points[:, 2] > xyz_min_max[2][0], points[:, 2] < xyz_min_max[2][1])
        mask = torch.logical_and(torch.logical_and(mask1, mask2), mask3)
    else:
        raise NotImplementedError
    filtered_points = points[mask]
    return filtered_points, mask


def pc_bbdx_filter(points, bbox_corners):
    """
    :para points: [N, 3 + K]
    :para bbox_corners: [8, 3], bounding box corners.
    """
    if isinstance(points, np.ndarray):
        min_bound = np.min(bbox_corners, axis=0)
        max_bound = np.max(bbox_corners, axis=0)
        inside_mask = np.all(np.logical_and(min_bound <= points, points <= max_bound), axis=1)
    elif isinstance(points, torch.Tensor):
        min_bound, _ = torch.min(bbox_corners, dim=0)
        max_bound, _ = torch.max(bbox_corners, dim=0)
        inside_mask = torch.all(torch.logical_and(min_bound <= points, points <= max_bound), dim=1)
    else:
        raise NotImplementedError
    filtered_points = points[inside_mask]
    return filtered_points, inside_mask


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


def xyz2uvz(pc_cam, intrinsic):
    pc_norm = np.dot(intrinsic, pc_cam.T).T
    pc_norm[:, 0] /= pc_norm[:, 2]
    pc_norm[:, 1] /= pc_norm[:, 2]
    # fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
    # u = fx * pc_cam[:, 0] / pc_cam[:, 2] + cx
    # v = fy * pc_cam[:, 1] / pc_cam[:, 2] + cy
    # uvz = np.concatenate((u[:, None], v[:, None], pc_cam[:, 2][:, None]), axis=1)
    return pc_norm


def uv2xyz(uvz, intrinsic):
    intrinsic_inv = np.linalg.inv(intrinsic)
    zuzvz = deepcopy(uvz)
    zuzvz[:, 0] *= zuzvz[:, 2]
    zuzvz[:, 1] *= zuzvz[:, 2]
    xyz = np.dot(intrinsic_inv, zuzvz.T).T
    return xyz
