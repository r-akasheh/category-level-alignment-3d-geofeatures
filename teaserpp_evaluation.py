import os
import pickle

import cv2
import requests
import roma
import torch
from scipy.spatial import cKDTree
import open3d as o3d
import numpy as np
from fastapi import HTTPException
from tqdm import tqdm

from main import registration_prep, extract_obj_pcd_from_scene, retrieve_obj_pcd
from mesh_transform import base_path
from model.lib.utils import to_o3d_pcd
from ransac_evaluation import retrieve_groundtruth_rotation, retrieve_groundtruth_translation

mappings = {}


def filter_items(file_path):
    """
    Reads a text file and returns the item names that start with "shoe", "cutlery", or "teapot".

    Parameters:
    file_path (str): The path to the text file.

    Returns:
    list: A list of item names that match the specified criteria.
    """
    # Define the prefixes to filter by
    prefixes = ["teapot", "shoe", "cutlery"]

    # Initialize an empty list to store matching item names
    matching_items = []

    try:
        # Open the file and read it line by line
        with open(file_path, 'r') as file:
            for line in file:
                # Split the line into parts (id, number, item name)
                parts = line.strip().split()

                # Ensure the line has the expected format
                if len(parts) != 3:
                    continue

                item_name = parts[2]

                # Check if the item name starts with any of the specified prefixes
                if any(item_name.startswith(prefix) for prefix in prefixes):
                    matching_items.append(item_name)

    except Exception as e:
        print(f"An error occurred: {e}")
        return []

    return matching_items


def nearest_neighbors_features(tgt_feats, src_feats, source, target):
    if (tgt_feats.cpu().numpy()).shape[0] < (src_feats.cpu().numpy()).shape[0]:
        less_feats, more_feats, src_reg_bool = [tgt_feats.cpu().numpy(), src_feats.cpu().numpy(), True]
    else:
        less_feats, more_feats, src_reg_bool = [src_feats.cpu().numpy(), tgt_feats.cpu().numpy(), False]

    pcd = source if src_reg_bool else target
    tree = cKDTree(more_feats)

    nearest_neighbors_indices = []
    for item in less_feats:
        result = tree.query(item, k=1)
        nearest_neighbors_indices.append(result[1])

    nearest_neighbors0 = pcd[nearest_neighbors_indices]
    nearest_neighbors = o3d.geometry.PointCloud()
    nearest_neighbors.points = o3d.utility.Vector3dVector(nearest_neighbors0.cpu().numpy())

    # True: Teaser reg with source
    return nearest_neighbors, src_reg_bool

def nearest_neighbors_features_old(tgt_feats, src_feats, source, target):
    if (tgt_feats.cpu().numpy()).shape[0] < (src_feats.cpu().numpy()).shape[0]:
        less_feats, more_feats, src_reg_bool = [tgt_feats.cpu().numpy(), src_feats.cpu().numpy(), True]
    else:
        less_feats, more_feats, src_reg_bool = [src_feats.cpu().numpy(), tgt_feats.cpu().numpy(), False]

    pcd = target if src_reg_bool else source
    tree = cKDTree(less_feats)

    nearest_neighbors_indices = []
    for item in more_feats:
        result = tree.query(item, k=1)
        nearest_neighbors_indices.append(result[1])

    nearest_neighbors0 = pcd[nearest_neighbors_indices]
    nearest_neighbors = o3d.geometry.PointCloud()
    nearest_neighbors.points = o3d.utility.Vector3dVector(nearest_neighbors0.cpu().numpy())

    # True: Teaser reg with source
    return nearest_neighbors, src_reg_bool

def nearest_neighbors_features_ratio_test(tgt_feats, src_feats, tgt_pcd, src_pcd, threshold: float = 0.8):
    flann_index_kdtree = 1
    index_params = dict(algorithm=flann_index_kdtree, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    if (tgt_feats.cpu().numpy()).shape[0] < (src_feats.cpu().numpy()).shape[0]:
        [matches, src_reg_bool] = [
            flann.knnMatch((src_feats.cpu().numpy()).astype(np.float32), (tgt_feats.cpu().numpy()).astype(np.float32),
                           k=2), True]
    else:
        [matches, src_reg_bool] = [
            flann.knnMatch((tgt_feats.cpu().numpy()).astype(np.float32), (src_feats.cpu().numpy()).astype(np.float32),
                           k=2), False]

    pcd = tgt_pcd if src_reg_bool else src_pcd

    filtered_matches = [m for m, n in matches if (m.distance / n.distance) < threshold]

    nearest_neighbors_indices = []
    for match in filtered_matches:
        query_idx, train_idx = match[0].queryIdx, match[0].trainIdx
        nearest_neighbors_indices.append(train_idx)

    nearest_neighbors0 = pcd[nearest_neighbors_indices]
    nearest_neighbors = o3d.geometry.PointCloud()
    nearest_neighbors.points = o3d.utility.Vector3dVector(nearest_neighbors0.cpu().numpy())

    return nearest_neighbors, src_reg_bool


def teaser(src_reg_bool, nearest_neighbors, source, target, visualize: bool = False):
    if ~src_reg_bool:
        # source_np = np.transpose(source.numpy())
        # target_np = np.transpose(np.asarray(nearest_neighbors.points))
        source = to_o3d_pcd(source)
        target = to_o3d_pcd(np.asarray(nearest_neighbors.points))

    else:
        # source_np = np.transpose(target.numpy())
        # target_np = np.transpose(np.asarray(nearest_neighbors.points))
        source = to_o3d_pcd(target)
        target = to_o3d_pcd(np.asarray(nearest_neighbors.points))

    if visualize:
        o3d.visualization.draw_geometries([source, target])

    o3d.io.write_point_cloud(filename=base_path + "TEASER-plusplus-docker/data/src_pcd.ply", pointcloud=source)
    o3d.io.write_point_cloud(filename=base_path + "TEASER-plusplus-docker/data/trg_pcd.ply", pointcloud=target)

    try:
        response = requests.post("http://localhost:8000/evaluate")
        rotation_matrix_str = response.json()["results"][0]
        translation_vector_str = response.json()["results"][1]
        rotation_matrix_cleaned = rotation_matrix_str.replace('\n', '').replace('[', '').replace(']', '').strip()
        rotation_matrix = np.fromstring(rotation_matrix_cleaned, sep=' ').reshape(3, 3)

        # For the rotation vector
        translation_vector_cleaned = translation_vector_str.replace('\n', '').replace('[', '').replace(']', '').strip()
        translation_vector = np.fromstring(translation_vector_cleaned, sep=' ')

        return rotation_matrix, translation_vector

    except HTTPException:
        return "Segmentation fault caught"


def run_teaser_eval(in_nocs_space: bool = False):
    for i in tqdm(range(1, 11)):
        print("ITERATION " + str(i))
        scene_nr_short = str(i).zfill(2)
        scene_nr = "scene" + scene_nr_short
        mappings[scene_nr] = {}
        scene_path = base_path + "scene1-10/{}/rgb".format(str(scene_nr))

        file_names = [f.split(".")[0] for f in os.listdir(scene_path)]
        for image_nr in tqdm(file_names):
            mappings[scene_nr][image_nr] = {}

            meta_path = base_path + "scene1-10/{}/meta.txt".format(str(scene_nr))
            matching_items = filter_items(meta_path)
            for item in matching_items:
                scene_pcd, obj_name, instance_id = extract_obj_pcd_from_scene(base_path,
                                                                              obj=item.split("-")[0],
                                                                              image_nr=image_nr,
                                                                              scene_nr=scene_nr_short)
                item_type = obj_name.split("-")[0]

                obj_pcd = retrieve_obj_pcd(obj_name)

                (n_sample_src, n_sample_trg, src_pcd, tgt_pcd,
                 src_feats, tgt_feats, point_cloud_src, point_cloud_trg) = registration_prep(scene_pcd, obj_pcd,
                                                                                             item_type)
                try:
                    if in_nocs_space:
                        nearest_neighbors, src_bool = nearest_neighbors_features(tgt_feats=tgt_feats,
                                                                                 src_feats=src_feats,
                                                                                 source=src_pcd,
                                                                                 target=tgt_pcd)
                        pred_rot, trans_pred = teaser(src_bool, nearest_neighbors, src_pcd, tgt_pcd)

                    else:
                        nearest_neighbors, src_bool = nearest_neighbors_features(tgt_feats=tgt_feats,
                                                                                 src_feats=src_feats,
                                                                                 source=src_pcd,
                                                                                 target=scene_pcd)
                        pred_rot, trans_pred = teaser(src_bool, nearest_neighbors, src_pcd, scene_pcd)

                    rot_pred = torch.tensor(pred_rot)

                    rot_ground_truth = retrieve_groundtruth_rotation(scene_nr_short, image_nr, instance_id)
                    trans_ground_truth = retrieve_groundtruth_translation(scene_nr_short, image_nr, instance_id)

                    theta = roma.rotmat_geodesic_distance(rot_ground_truth, rot_pred)  # In radian
                    cos_theta = roma.rotmat_cosine_angle(rot_ground_truth.transpose(-2, -1) @ rot_pred)

                    euclid_error = np.linalg.norm(trans_ground_truth - trans_pred)
                    mappings[scene_nr][image_nr][item_type] = {"pred_trans": trans_pred,
                                                               "cos_theta": cos_theta,
                                                               "euclid_error": euclid_error,
                                                               "obj_name": obj_name,
                                                               "theta": theta
                                                               }
                except:
                    mappings[scene_nr][image_nr][item_type] = {"pred_trans": "error",
                                                               "cos_theta": "error",
                                                               "euclid_error": "error",
                                                               "obj_name": "error",
                                                               "theta": "error"
                                                               }

    with open("evaluation/teaser_finetuned_nn_scene_full_fixed_prep.pickle", 'wb') as f:
        pickle.dump(mappings, f)


if __name__ == "__main__":
    run_teaser_eval()
