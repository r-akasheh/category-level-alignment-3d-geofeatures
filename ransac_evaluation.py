import os
import numpy as np
import torch
import pickle

from evaluation.utils import filter_items, retrieve_groundtruth_rotation, retrieve_groundtruth_translation
from main import extract_obj_pcd_from_scene, retrieve_obj_pcd, ransac, registration_prep
from mesh_transform import base_path

import roma
from tqdm import tqdm

mappings = {}

def run_ransac_evals():
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
                scene_pcd, obj_name, instance_id = extract_obj_pcd_from_scene(base_path, obj=item.split("-")[0],
                                                                              image_nr=image_nr,
                                                                              scene_nr=scene_nr_short)
                item_type = obj_name.split("-")[0]

                obj_pcd = retrieve_obj_pcd(obj_name)

                (n_sample_src, n_sample_trg, src_pcd, tgt_pcd,
                 src_feats, tgt_feats, point_cloud_src, point_cloud_trg) = registration_prep(scene_pcd, obj_pcd, item_type)

                pred_trans = ransac(n_sample_src, n_sample_trg, src_pcd, tgt_pcd,
                                    src_feats, tgt_feats, point_cloud_src, point_cloud_trg)

                rot_pred_transf = torch.tensor(pred_trans[:3, :3])
                trans_pred_transf = pred_trans[:3, 3]

                rot_ground_truth = retrieve_groundtruth_rotation(scene_nr, image_nr, instance_id)
                trans_ground_truth = retrieve_groundtruth_translation(scene_nr, image_nr, instance_id)

                theta = roma.rotmat_geodesic_distance(rot_ground_truth, rot_pred_transf)  # In radian
                cos_theta = roma.rotmat_cosine_angle(rot_ground_truth.transpose(-2, -1) @ rot_pred_transf)

                euclid_error = np.linalg.norm(trans_ground_truth - trans_pred_transf)
                mappings[scene_nr][image_nr][item_type] = {"pred_trans": pred_trans,
                                                           "theta": theta,
                                                           "cos_theta": cos_theta,
                                                           "euclid_error": euclid_error,
                                                           "obj_name": obj_name
                                                           }

    with open("evaluation/ransac_mappings_finetuned_projection.pickle", 'wb') as f:
        pickle.dump(mappings, f)

run_ransac_evals()
