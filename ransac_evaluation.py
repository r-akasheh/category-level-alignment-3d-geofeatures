import os
import numpy as np
import torch
import pickle

from main import extract_obj_pcd_from_scene, retrieve_obj_pcd, load_features_run_ransac
from mesh_transform import base_path

import roma
from tqdm import tqdm

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
    prefixes = ["shoe", "cutlery", "teapot"]

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


for i in tqdm(range(1, 11)):
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
            scene_pcd, obj_name = extract_obj_pcd_from_scene(base_path, obj=item.split("-")[0], image_nr=image_nr,
                                                             scene_nr=scene_nr_short)
            item_type = obj_name.split("-")[0]

            obj_pcd = retrieve_obj_pcd(obj_name)

            result_ransac, pred_trans = load_features_run_ransac(scene_pcd, obj_pcd, item_type)
            rot_pred_transf = torch.tensor(pred_trans[:3, :3])
            trans_pred_transf = pred_trans[:3, 3]

            ground_truth = torch.tensor(np.loadtxt(base_path + "scene1-10/{}/obj_pose_final/{}".format(str(scene_nr),
                                                                                                       obj_name + ".txt")))
            rot_ground_truth = torch.tensor(ground_truth[:3, :3])
            trans_ground_truth = ground_truth[:3, 3]

            theta = roma.rotmat_geodesic_distance(rot_ground_truth, rot_pred_transf)  # In radian
            cos_theta = roma.rotmat_cosine_angle(rot_ground_truth.transpose(-2, -1) @ rot_pred_transf)

            euclid_error = np.linalg.norm(trans_ground_truth - trans_pred_transf)
            mappings[scene_nr][image_nr][item_type] = {"pred_trans": pred_trans,
                                                       "cos_theta": cos_theta,
                                                       "euclid_error": euclid_error,
                                                       "obj_name": obj_name
                                                       }

with open("evaluation/ransac_mappings.pickle", 'wb') as f:
    pickle.dump(mappings, f)
