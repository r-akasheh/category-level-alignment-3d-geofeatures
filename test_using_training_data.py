import pickle

import roma
import torch
from tqdm import tqdm

from main import registration_prep, ransac
from mesh_transform import base_path
import os
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

from teaserpp_evaluation import nearest_neighbors_features, teaser


# load in a complete/incomplete point cloud

def random_rotation_matrix():
    # Generate a random rotation matrix using scipy
    random_rotation = R.random().as_matrix()
    return random_rotation


def random_translation_vector():
    # Generate a random translation vector
    translation_vector = np.random.uniform(-1, 1, size=3)
    return translation_vector


# Generate random rotation matrix and translation vector
rotation_matrix = random_rotation_matrix()
translation_vector = random_translation_vector()

# Create a 4x4 transformation matrix
transformation_matrix = np.eye(4)
transformation_matrix[:3, :3] = rotation_matrix
transformation_matrix[:3, 3] = translation_vector

item_types = ["cutlery", "shoe", "teapot"]
mappings_ransac = {}
mappings_teaser = {}
def eval_ransac_objects():
    for item in item_types:
        path_to_npz_files = base_path + "category-level-alignment-3d-geofeatures/mesh_transform/npz_files/" + item + "/"
        for file in tqdm(os.listdir(path_to_npz_files)):
            # load in file
            data = np.load(path_to_npz_files + file)
            points = data['pcd']  # Assuming 'points' is the key for the point cloud data
            obj_name = file.split(".")[0]
            # Create an Open3D point cloud object
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(points)
            # apply a random transformation (rot and trans)

            point_cloud_transformed = point_cloud.transform(transformation_matrix)

            n_sample_src, n_sample_trg, src_pcd, tgt_pcd, src_feats, tgt_feats, point_cloud_src, point_cloud_trg = (
                registration_prep(obj_pcd=point_cloud, scene_pcd=point_cloud_transformed, item_type=item))

            pred_trans = ransac(n_sample_src, n_sample_trg, src_pcd, tgt_pcd,
                                src_feats, tgt_feats, point_cloud_src, point_cloud_trg)

            rot_pred_transf = torch.tensor(pred_trans[:3, :3])
            trans_pred_transf = pred_trans[:3, 3]
            theta = roma.rotmat_geodesic_distance(torch.tensor(rotation_matrix), rot_pred_transf)  # In radian
            cos_theta = roma.rotmat_cosine_angle(torch.tensor(rotation_matrix).transpose(-2, -1) @ rot_pred_transf)

            euclid_error = np.linalg.norm(translation_vector - trans_pred_transf)

            mappings_ransac[obj_name] = {"pred_trans": pred_trans,
                                         "theta": theta,
                                         "cos_theta": cos_theta,
                                         "euclid_error": euclid_error,
                                         "obj_name": obj_name
                                         }

    with open("evaluation/ransac_mappings_objects_baseline_2.pickle", 'wb') as f:
        pickle.dump(mappings_ransac, f)

def eval_teaser_objects():
    for item in item_types:
        path_to_npz_files = base_path + "category-level-alignment-3d-geofeatures/mesh_transform/npz_files/" + item + "/"
        for file in tqdm(os.listdir(path_to_npz_files)):
            # load in file
            data = np.load(path_to_npz_files + file)
            points = data['pcd']  # Assuming 'points' is the key for the point cloud data
            obj_name = file.split(".")[0]
            # Create an Open3D point cloud object
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(points)
            # apply a random transformation (rot and trans)

            point_cloud_transformed = point_cloud.transform(transformation_matrix)

            n_sample_src, n_sample_trg, src_pcd, tgt_pcd, src_feats, tgt_feats, point_cloud_src, point_cloud_trg = (
                registration_prep(obj_pcd=point_cloud, scene_pcd=point_cloud_transformed, item_type=item))

            nearest_neighbors, src_bool = nearest_neighbors_features(tgt_feats, src_feats, src_pcd, tgt_pcd)

            pred_rot, trans_pred = teaser(src_bool, nearest_neighbors, src_pcd, tgt_pcd)

            rot_pred_transf = torch.tensor(pred_rot)
            trans_pred_transf = trans_pred

            theta = roma.rotmat_geodesic_distance(torch.tensor(rotation_matrix), rot_pred_transf)  # In radian
            cos_theta = roma.rotmat_cosine_angle(torch.tensor(rotation_matrix).transpose(-2, -1) @ rot_pred_transf)

            euclid_error = np.linalg.norm(translation_vector - trans_pred_transf)

            mappings_teaser[obj_name] = {"theta": theta,
                                         "cos_theta": cos_theta,
                                         "euclid_error": euclid_error,
                                         "obj_name": obj_name
                                         }

    with open("evaluation/teaser_mappings_objects_finetuned.pickle", 'wb') as f:
        pickle.dump(mappings_teaser, f)


eval_ransac_objects()


