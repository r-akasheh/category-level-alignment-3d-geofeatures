import numpy as np
import roma

from evaluation.utils import retrieve_groundtruth_rotation, retrieve_groundtruth_translation
from main import extract_obj_pcd_from_scene, retrieve_obj_pcd, registration_prep
from mesh_transform import base_path

from teaserpp_evaluation import nearest_neighbors_features, teaser
import torch

from utils.ransac_utils import visualize_registration

scene_nr = "06"
image_nr = "000288"
obj = "shoe"
scene_pcd, obj_name, instance_id = extract_obj_pcd_from_scene(base_path, obj=obj,
                                                              image_nr=image_nr,
                                                              scene_nr=scene_nr)
item_type = obj_name.split("-")[0]

obj_pcd = retrieve_obj_pcd(obj_name)

n_sample_src, n_sample_trg, src_pcd, tgt_pcd, src_feats, tgt_feats, point_cloud_src, point_cloud_trg = (
    registration_prep(scene_pcd=scene_pcd, obj_pcd=obj_pcd, item_type=item_type))

nearest_neighbors, src_bool = nearest_neighbors_features(tgt_feats=tgt_feats,
                                                         src_feats=src_feats,
                                                         target=tgt_pcd,
                                                         source=src_pcd)
pred_rot, pred_trans = teaser(src_reg_bool=src_bool, nearest_neighbors=nearest_neighbors,
                              source=src_pcd, target=tgt_pcd)

rot_pred = torch.tensor(pred_rot)
rot_ground_truth = retrieve_groundtruth_rotation(scene_nr, image_nr, instance_id)
trans_ground_truth = retrieve_groundtruth_translation(scene_nr, image_nr, instance_id)

theta = roma.rotmat_geodesic_distance(rot_ground_truth, rot_pred)  # In radian
cos_theta = roma.rotmat_cosine_angle(rot_ground_truth.transpose(-2, -1) @ rot_pred)

euclid_error = np.linalg.norm(trans_ground_truth - pred_trans)
print(cos_theta)


def combine_rotation_translation(rotation_matrix, translation_vector):
    # Create a 4x4 identity matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = translation_vector

    return transformation_matrix


#visualize_registration(src_ply=point_cloud_src, tgt_ply=point_cloud_trg, pred_trans=combine_rotation_translation(pred_rot, pred_trans))
print(combine_rotation_translation(pred_rot, pred_trans))