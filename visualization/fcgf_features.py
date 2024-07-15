import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from main import extract_obj_pcd_from_scene, retrieve_obj_pcd, registration_prep
from mesh_transform import base_path


def feature_visualizer(src_pcd, src_feats, tgt_pcd, tgt_feats):
    src_feats = src_feats.cpu()
    tgt_feats = tgt_feats.cpu()
    src_pcd = src_pcd.cpu()
    tgt_pcd = tgt_pcd.cpu()

    ## GPT CODE
    scaler_src = StandardScaler()
    src_feats_scaled = scaler_src.fit_transform(src_feats)

    scaler_trg = StandardScaler()
    trg_feats_scaled = scaler_trg.fit_transform(tgt_feats)

    # Apply PCA to reduce to 3 dimensions for each dataset
    pca_src = PCA(n_components=3)
    src_pca = pca_src.fit_transform(src_feats_scaled)

    pca_trg = PCA(n_components=3)
    trg_pca = pca_trg.fit_transform(trg_feats_scaled)

    # Normalize the PCA components to the range [0, 1]
    src_pca_min, src_pca_max = src_pca.min(axis=0), src_pca.max(axis=0)
    trg_pca_min, trg_pca_max = trg_pca.min(axis=0), trg_pca.max(axis=0)

    src_pca_normalized = (src_pca - src_pca_min) / (src_pca_max - src_pca_min)
    trg_pca_normalized = (trg_pca - trg_pca_min) / (trg_pca_max - trg_pca_min)

    ## GPT CODE END

    pcd_src = o3d.geometry.PointCloud()
    pcd_src.points = o3d.utility.Vector3dVector(src_pcd.numpy())
    pcd_src.colors = o3d.utility.Vector3dVector(src_pca_normalized)

    pcd_trg = o3d.geometry.PointCloud()
    pcd_trg.points = o3d.utility.Vector3dVector(tgt_pcd.numpy())
    pcd_trg.colors = o3d.utility.Vector3dVector(trg_pca_normalized)
    pcd_trg.translate((1, 0, 0))
    o3d.visualization.draw_geometries([pcd_src, pcd_trg])


obj_pcd_src = retrieve_obj_pcd("shoe-crocs_white_cyan_right")
obj_pcd_trg = retrieve_obj_pcd("shoe-magenta_holes_right")


_, _, src_pcd, tgt_pcd, src_feats, trg_feats, _, _ = registration_prep(obj_pcd_src, obj_pcd_trg, "shoe",
                                                                       use_baseline=False)

feature_visualizer(src_pcd=src_pcd, tgt_pcd=tgt_pcd, src_feats=src_feats, tgt_feats=trg_feats)



