from scipy.spatial import cKDTree
import open3d as o3d


def nearest_neighbors_features(tgt_feats, src_feats, source, target):

    if (tgt_feats.numpy()).shape[0] < (src_feats.numpy()).shape[0]:
        less_feats, more_feats, src_reg_bool = [tgt_feats.numpy(), src_feats.numpy(), True]
    else:
        less_feats, more_feats, src_reg_bool = [src_feats.numpy(), tgt_feats.numpy(), False]

    pcd = target if src_reg_bool else source
    tree = cKDTree(less_feats.numpy())

    nearest_neighbors_indices = []
    for item in more_feats.numpy():
        result = tree.query(item, k=1)
        nearest_neighbors_indices.append(result[1])

    nearest_neighbors0 = (pcd.numpy())[nearest_neighbors_indices]
    nearest_neighbors = o3d.geometry.PointCloud()
    nearest_neighbors.points = o3d.utility.Vector3dVector(nearest_neighbors0)

    # True: Teaser reg with source
    return nearest_neighbors, src_reg_bool
