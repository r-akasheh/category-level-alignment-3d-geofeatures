import numpy as np
import open3d as o3d
from sklearn.decomposition import PCA

from main import extract_obj_pcd_from_scene, retrieve_obj_pcd, registration_prep
from mesh_transform import base_path


def feature_visualizer(src_pcd, src_feats, tgt_pcd, tgt_feats):
    src_feats = src_feats.cpu()
    tgt_feats = tgt_feats.cpu()
    src_pcd = src_pcd.cpu()
    tgt_pcd = tgt_pcd.cpu()
    tsne = PCA(n_components=3).fit_transform(src_feats)
    if np.min(tsne) < 0:
        tsne = tsne - np.min(tsne)

    pcd0 = o3d.geometry.PointCloud()
    pcd0.points = o3d.utility.Vector3dVector(src_pcd.numpy())
    pcd0.colors = o3d.utility.Vector3dVector((tsne) / np.max(tsne))
    o3d.visualization.draw_geometries([pcd0])

    tsne = PCA(n_components=3).fit_transform(tgt_feats)
    if np.min(tsne) < 0:
        tsne = tsne - np.min(tsne)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(tgt_pcd.numpy())
    pcd.colors = o3d.utility.Vector3dVector(tsne / np.max(tsne))
    pcd.translate((1, 0, 0))
    o3d.visualization.draw_geometries([pcd, pcd0])


scene_pcd, obj_name = extract_obj_pcd_from_scene(base_path, obj="shoe", image_nr="000113",
                                                                 scene_nr="07")
item_type = obj_name.split("-")[0]
obj_pcd = retrieve_obj_pcd(obj_name)

_, _, src_pcd, tgt_pcd, src_feats, tgt_feats, _, _ = registration_prep(scene_pcd, obj_pcd, item_type)
feature_visualizer(src_pcd, src_feats, tgt_pcd, tgt_feats)
