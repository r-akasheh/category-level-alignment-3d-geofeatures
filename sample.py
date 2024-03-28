from augmentation.augment import augmenting_the_mesh, augmenting_from_depth
from matching_pairs.matching import extract_matching_pairs
from mesh_transform import base_path
from mesh_transform.mesh2pcd import obj_to_point_cloud
from mesh_transform.scale_shift import scale_shift
import numpy as np
import open3d as o3d

from mesh_transform.transform_depth_rgbd import transform_depth_rgbd_to_pcd

# Provide the path to your 3D mesh file
mesh_file_path = base_path + "obj_models/obj_models_small_size_final/shoe/shoe-skyblue_leda_right.obj"
point_cloud = obj_to_point_cloud(mesh_file_path)
x = scale_shift(point_cloud, True)

mesh_file_path = base_path + "obj_models/obj_models_small_size_final/shoe/shoe-hummel_green_sandal_right.obj"
point_cloud = obj_to_point_cloud(mesh_file_path)
y = scale_shift(point_cloud, False)

# positive_pairs, negative_pairs = extract_matching_pairs(x, y, 0.0005)
augmented_pcd_1 = augmenting_the_mesh(mesh_file_path, view_direction=np.array([3, 3, 3]), visualize=True)

augmented_pcd_2 = augmenting_from_depth(mesh_file_path, visualize=True)

transform_depth_rgbd_to_pcd(visualize=True)