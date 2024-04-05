import os
import numpy as np
from mesh2pcd import obj_to_point_cloud
from mesh_transform.scale_shift import scale_shift


def npz_generator(object_type: str):

    path = "obj_models_small_size_final/"

    directory_path = path + object_type + "/"
    for filename in os.listdir(directory_path):
        if filename.endswith(".obj"):
            point_cloud = obj_to_point_cloud(directory_path + filename)
            point_cloud = scale_shift(point_cloud)
            pts = np.array(point_cloud.points)
            color = np.zeros(pts.shape)
            saving_file = "zipped_files/" + object_type + "/"
            np.savez(saving_file + filename[:-4], pcd=pts, color=color)
