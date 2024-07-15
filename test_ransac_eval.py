import numpy as np
import roma
import torch
import copy

from evaluation.utils import retrieve_groundtruth_rotation, retrieve_groundtruth_translation
from main import extract_obj_pcd_from_scene, retrieve_obj_pcd, registration_prep, ransac
from mesh_transform import base_path

image_nr = "000025"
scene_nr = "07"
obj = "teapot"
scene_pcd, obj_name, instance_id = extract_obj_pcd_from_scene(base_path,
                                                              obj=obj,
                                                              image_nr=image_nr,
                                                              scene_nr=scene_nr)
item_type = obj_name.split("-")[0]

obj_pcd = copy.deepcopy(retrieve_obj_pcd(obj_name))

(n_sample_src, n_sample_trg, src_pcd, tgt_pcd,
 src_feats, tgt_feats, _, _) = registration_prep(scene_pcd, obj_pcd,
                                                 item_type)

pred_trans = ransac(n_sample_src=n_sample_src,
                    n_sample_trg=n_sample_trg,
                    src_pcd=src_pcd,
                    tgt_pcd=tgt_pcd,
                    src_feats=src_feats,
                    tgt_feats=tgt_feats,
                    point_cloud_src=obj_pcd,
                    point_cloud_trg=scene_pcd,
                    visualize=True)

rot_pred_transf = torch.tensor(pred_trans[:3, :3])
trans_pred_transf = pred_trans[:3, 3]

rot_ground_truth = retrieve_groundtruth_rotation(scene_nr, image_nr, instance_id)
trans_ground_truth = retrieve_groundtruth_translation(scene_nr, image_nr, instance_id)

theta = roma.rotmat_geodesic_distance(rot_ground_truth, rot_pred_transf)  # In radian
cos_theta = roma.rotmat_cosine_angle(rot_ground_truth.transpose(-2, -1) @ rot_pred_transf)

euclid_error = np.linalg.norm(trans_ground_truth - trans_pred_transf)

print(cos_theta)
print(theta)
