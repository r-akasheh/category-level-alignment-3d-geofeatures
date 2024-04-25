import os

import numpy as np

from augmentation.augment import augmenting_the_mesh, augmenting_from_depth
from mesh_transform import base_path
from mesh_transform.mesh2pcd import obj_to_point_cloud
from mesh_transform.scale_shift import scale_shift, scale_shift_nocs_color_demonstration
from mesh_transform.transform_depth_rgbd import transform_depth_rgbd_to_pcd

# Provide the path to your 3D mesh file
mesh_file_path = base_path + "obj_models/obj_models_small_size_final/shoe/shoe-skyblue_leda_right.obj"
point_cloud = obj_to_point_cloud(mesh_file_path)
x = scale_shift(point_cloud, True)

mesh_file_path = base_path + "obj_models/obj_models_small_size_final/shoe/shoe-hummel_green_sandal_right.obj"
point_cloud = obj_to_point_cloud(mesh_file_path)
y = scale_shift(point_cloud, False)

# positive_pairs, negative_pairs = extract_matching_pairs(x, y, 0.0005)

# Show an augmentation example to produce an incomplete point cloud by the first method
augmented_pcd_1 = augmenting_the_mesh(mesh_file_path, view_direction=np.array([3, 3, 3]), visualize=True)

# Show an augmentation example to produce an incomplete point cloud by the second method
augmented_pcd_2 = augmenting_from_depth(mesh_file_path, visualize=True)

# Show how an RGBD and depth image can be merged into a point cloud
transform_depth_rgbd_to_pcd(visualize=True)

# Show the color distribution of a sample PCD and overlaying points

object_type = "shoe"

# Shoes
array_test = np.array([[0.5,0.5,0.5], [0.1, 0.5, 0.5], [0.9, 0.5, 0.5], [0.5,0.7,0.5], [0.5,0.6,0.64]])
# Knife
# array_test = np.array([[0.5,0.5,0.5], [0.1, 0.5, 0.5], [0.9, 0.5, 0.5]])
# Fork and spoon
# array_test = np.array([[0.5,0.5,0.5], [0.1, 0.5, 0.5], [0.9, 0.55, 0.5]])
# Remote
# array_test = np.array([[0.5,0.55,0.64], [0.1, 0.55, 0.36], [0.9, 0.55, 0.5], [0.5,0.45,0.36], [0.5,0.55,0.36]])
# Teapots!!!! Problem
# array_test = np.array([[0.5,0.55,0.64], [0.1, 0.55, 0.5], [0.9, 0.55, 0.5], [0.5,0.45,0.36], [0.5,0.55,0.36]])
# Teapots!!!!
# array_test = np.array([[0.5,0.1,0.64], [0.45, 0.9, 0.5], [0.43, 0.7, 0.5], [0.5,0.45,0.65], [0.5,0.1,0.36]])


directory_path = base_path + "obj_models/obj_models_small_size_final/" + object_type + "/"
for filename in os.listdir(directory_path):
    if filename.endswith(".obj"):
        point_cloud = obj_to_point_cloud(directory_path + filename)
        y = scale_shift_nocs_color_demonstration(point_cloud, array_test, True)

