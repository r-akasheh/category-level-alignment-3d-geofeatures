import os

import numpy as np

#from augmentation.augment import augmenting_the_mesh, augmenting_from_depth, random_remove_points_with_spared_radius, \
#    extract_spherical_region, extract_bounding_box_region
#from main import registration_prep, extract_obj_pcd_from_scene, retrieve_obj_pcd, ransac
#from mesh_transform import base_path
#from mesh_transform.mesh2pcd import obj_to_point_cloud
#from mesh_transform.scale_shift import scale_shift, scale_shift_nocs_color_demonstration
#from mesh_transform.transform_depth_rgbd import transform_depth_rgbd_to_pcd
#from teaserpp_evaluation import nearest_neighbors_features, teaser

# Provide the path to your 3D mesh file
# mesh_file_path = base_path + "obj_models/obj_models_small_size_final/teapot/teapot-white_small.obj"
# point_cloud = obj_to_point_cloud(mesh_file_path)
# x = scale_shift(point_cloud, False)

# random_remove_points_with_spared_radius(point_cloud=x, removal_fraction=0.9, spared_radius=1, visualize=True)
# random_remove_points_with_spared_radius(point_cloud=x, removal_fraction=0.7, spared_radius=0.04, visualize=True,
#                                        center_point=(0.2, 0.2, 0.2))

# mesh_file_path = base_path + "obj_models/obj_models_small_size_final/shoe/shoe-hummel_green_sandal_right.obj"
# point_cloud = obj_to_point_cloud(mesh_file_path)
# y = scale_shift(point_cloud, False)

# Show an augmentation example to produce an incomplete point cloud by the first method
# augmented_pcd_1 = augmenting_the_mesh(mesh_file_path, view_direction=np.array([3, 3, 3]), visualize=True)
# random_remove_points_with_spared_radius(point_cloud=augmented_pcd_1, removal_fraction=0.96, spared_radius=0.01,
#                                        visualize=True)
# extract_bounding_box_region(point_cloud=augmented_pcd_1, center=(0.5,0.5,0.5), size=15, visualize=True)
# extract_spherical_region(point_cloud=augmented_pcd_1, center=(0, 0, 0), radius=0.5, visualize=True)

# Show an augmentation example to produce an incomplete point cloud by the second method
# augmented_pcd_2 = augmenting_from_depth(mesh_file_path, visualize=True)

# Show how an RGBD and depth image can be merged into a point cloud
# transform_depth_rgbd_to_pcd(visualize=True)

# Show the color distribution of a sample PCD and overlaying points

# object_type = "shoe"

# Shoes
# array_test = np.array([[0.5,0.5,0.5], [0.1, 0.5, 0.5], [0.9, 0.5, 0.5], [0.5,0.7,0.5], [0.5,0.6,0.64]])
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


# directory_path = base_path + "obj_models/obj_models_small_size_final/" + object_type + "/"
# for filename in os.listdir(directory_path):
#    if filename.endswith(".obj"):
#        point_cloud = obj_to_point_cloud(directory_path + filename)
#        y = scale_shift_nocs_color_demonstration(point_cloud, array_test, True)


# pcd = transform_depth_rgbd_to_pcd(scene="scene07", visualize=True, object_type="teapot", mask=True,
#                                  obj_name="teapot-blue_floral")

# scene_pcd, obj_name, instance_id = extract_obj_pcd_from_scene(base_path, obj="shoe", image_nr="000243",
#                                                 scene_nr="07", visualize=True)
# item_type = obj_name.split("-")[0]#

# obj_pcd = retrieve_obj_pcd(obj_name)

# (n_sample_src, n_sample_trg, src_pcd, tgt_pcd,
# src_feats, tgt_feats, point_cloud_src, point_cloud_trg) = registration_prep(scene_pcd, obj_pcd, item_type)

# pred_trans = ransac(n_sample_src, n_sample_trg, src_pcd, tgt_pcd,
#                    src_feats, tgt_feats, point_cloud_src, point_cloud_trg, visualize=True)

# nearest_neighbors, src_bool = nearest_neighbors_features(tgt_feats, src_feats, src_pcd, tgt_pcd)
# pred_rot, trans_pred = teaser(src_bool, nearest_neighbors, src_pcd, tgt_pcd)
# print(pred_rot, trans_pred)
# print(obj_name)


