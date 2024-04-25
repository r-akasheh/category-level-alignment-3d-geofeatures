import os
import numpy as np

from augmentation.augment import augmenting_the_mesh
from mesh2pcd import obj_to_point_cloud
from mesh_transform import base_path
from mesh_transform.scale_shift import scale_shift

from scipy.spatial import KDTree


def chamfer_distance(pcd_a, pcd_b):
    """
    Computes the chamfer distance between two sets of points A and B.
    """
    tree = KDTree(pcd_b)
    dist_a = tree.query(pcd_a)[0]
    tree = KDTree(pcd_a)
    dist_b = tree.query(pcd_b)[0]
    return np.mean(dist_a) + np.mean(dist_b)


def npz_generator(object_type: str, viewing_directions: list):
    path = "obj_models/obj_models_small_size_final/"
    names = []
    directory_path = base_path + path + object_type + "/"
    for filename in os.listdir(directory_path):
        if filename.endswith(".obj"):
            obj_name = filename.split(".")[0]
            point_cloud = obj_to_point_cloud(directory_path + filename)
            point_cloud = scale_shift(point_cloud)
            pts = np.array(point_cloud.points)
            color = np.zeros(pts.shape)
            saving_file = "npz_files/"
            np.savez(saving_file + obj_name, pcd=pts, color=color)
            for i, viewing_direction in enumerate(viewing_directions):
                pcd = augmenting_the_mesh(directory_path + filename, viewing_direction)
                pts = np.array(pcd.points)
                color = np.zeros(pts.shape)
                saving_file = "npz_files/"
                np.savez(saving_file + obj_name + "00" + str(i), pcd=pts, color=color)


item_types = ["shoe", "teapot", "cutlery"]
viewing_directions = [[0, 5, 1], [2, 2, 0], [-1, -2, 0],
                      [1, 0, 2], [0, 3, 7], [-2, 4, -1],
                      [5, 0, 1], [1, 1, 1], [1, 6, 3]]
for item in item_types:
    npz_generator(object_type=item, viewing_directions=viewing_directions)
