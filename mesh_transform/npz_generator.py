import os
import numpy as np
from mesh2pcd import obj_to_point_cloud


def npz_generator(object: str):

    path = "obj_models_small_size_final/"

    directory_path = path + object + "/"
    for filename in os.listdir(directory_path):
        if filename.endswith(".obj"):
            point_cloud = obj_to_point_cloud(directory_path + filename)
            pts = np.array(point_cloud.points)  # done
            color = np.zeros(pts.shape)
            saving_file = "zipped_files/" + object + "/"
            np.savez(saving_file + filename[:-4], pcd=pts, color=color)


