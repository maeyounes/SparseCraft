from plyfile import PlyData
import numpy as np
import cv2


def ply_to_point_cloud(ply_file, radius=None):
    ply_data = PlyData.read(ply_file)

    # Get vertex coordinates from the PLY file
    x = ply_data["vertex"]["x"]
    y = ply_data["vertex"]["y"]
    z = ply_data["vertex"]["z"]

    # Get vertex normals from the PLY file
    nx = ply_data["vertex"]["nx"]
    ny = ply_data["vertex"]["ny"]
    nz = ply_data["vertex"]["nz"]

    # Get vertex colors from the PLY file
    red = ply_data["vertex"]["red"]
    green = ply_data["vertex"]["green"]
    blue = ply_data["vertex"]["blue"]

    np.column_stack((x, y, z, red, green, blue))
    # return points, colors and normals
    pcl_points = np.column_stack((x, y, z))
    pcl_normals = np.column_stack((nx, ny, nz))
    pcl_colors = np.column_stack((red, green, blue)) / 255.0
    
    if radius:
        # Euclidean distances from the origin
        distances = np.linalg.norm(pcl_points, axis=1)
        radius_mask = (distances >= -radius) & (distances <= radius)
        pcl_points = pcl_points[radius_mask]
        pcl_normals = pcl_normals[radius_mask]
        pcl_colors = pcl_colors[radius_mask]    
    return pcl_points, pcl_colors, pcl_normals


def load_K_Rt_from_P(P=None):
    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose
