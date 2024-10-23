from typing import Dict, List, Union
from trimesh.base import Trimesh
from open3d.geometry import Geometry, PointCloud, TriangleMesh, LineSet, AxisAlignedBoundingBox
from open3d.utility import Vector3dVector
from open3d.visualization import draw_geometries
import numpy as np
from numpy import ndarray


def draw_o3d_geometries(geos: List[Union[Geometry, ndarray]], draw_frame=True):
    '''
    points: (N, 3+3+C): xyz, rgb(0~1), feats
    '''
    draw_geos = []
    for geo in geos:
        if type(geo) is ndarray:
            assert len(geo.shape) == 2 and geo.shape[1] >= 3, 'numpy.ndarray shape must be (N, 3+)'
            pcd = PointCloud()
            pcd.points = Vector3dVector(geo[:, :3])
            if geo.shape[1] > 3:
                pcd.colors = Vector3dVector(geo[:, 3:6])
            draw_geos.append(pcd)
        else:
            assert isinstance(geo, (PointCloud, TriangleMesh, LineSet)), "Each element in the list should be a valid geometry object."
            draw_geos.append(geo)
    if draw_frame:
        draw_geos.append(TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0]))
    draw_geometries(draw_geos)


def np2pcd(points, colors=None, normals=None):
    """Convert numpy array to open3d PointCloud."""
    pc = PointCloud()
    pc.points = Vector3dVector(points.copy())
    if colors is not None:
        colors = np.array(colors)
        if colors.ndim == 2:
            assert len(colors) == len(points)
        elif colors.ndim == 1:
            colors = np.tile(colors, (len(points), 1))
        else:
            raise RuntimeError(colors.shape)
        pc.colors = Vector3dVector(colors)
    if normals is not None:
        assert len(points) == len(normals)
        pc.normals = Vector3dVector(normals)
    return pc


def crop_pcd(pcd: PointCloud, bounds_list):
    """
    crop pcd according to bounds_list.
    bounds_list: [3, 2] defines the range of xyz.
    """
    bbox_pt_vec = Vector3dVector(bounds_list)
    bbox = AxisAlignedBoundingBox.create_from_points(bbox_pt_vec)
    pcd_cropped = pcd.crop(bbox)
    # draw_geometries([pcd])
    # draw_geometries([pcd_cropped])
    return pcd_cropped


def remove_plane(pcd: PointCloud, dist_th=0.01, ransac_n=3, num_it=1000):
    """
        identify the plane and remove it
    """
    plane_model, inliers = pcd.segment_plane(distance_threshold=dist_th, ransac_n=ransac_n, num_iterations=num_it)
    plane_cloud = pcd.select_by_index(inliers)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    outlier_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    ## debug
    # plane_cloud.paint_uniform_color([1, 0, 0])
    # outlier_cloud.paint_uniform_color([0, 0, 1])
    # draw_geometries([plane_cloud, outlier_cloud])
    return outlier_cloud, plane_cloud
