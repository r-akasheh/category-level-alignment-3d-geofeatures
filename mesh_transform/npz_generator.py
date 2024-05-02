import os
import random
import pickle
import numpy as np
import re

from augmentation.augment import augmenting_the_mesh
from mesh2pcd import obj_to_point_cloud
from mesh_transform import base_path
from mesh_transform.scale_shift import scale_shift

from scipy.spatial import KDTree
import open3d as o3d


def remove_last_three_digits(string):
    # Define a regular expression pattern to match the last three digits
    pattern = r'\d{3}$'
    # Use re.sub() to replace the matched pattern with an empty string
    new_string = re.sub(pattern, '', string)
    return new_string


def load_pcd_from_file(npz_name):
    path_to_npz = (base_path + "category-level-alignment-3d-geofeatures/mesh_transform/npz_files/" +
                   npz_name.split("-")[0] + "/" + npz_name)

    # Load the point cloud data from the npz file
    data = np.load(path_to_npz)
    points = data['pcd']  # Assuming 'points' is the key for the point cloud data

    # Create an Open3D point cloud object
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    return points


def get_random_file_pairs(directory, obj_name, num_pairs=1):
    # Get a list of all files in the directory
    all_files = os.listdir(directory)

    # Filter out directories from the list
    files = [file for file in all_files if obj_name == remove_last_three_digits(file.split(".")[0])]

    # Shuffle the list of files
    random.seed(42)
    random.shuffle(files)

    # Generate pairs
    random_pairs = []
    for i in range(min(num_pairs, len(files) // 2)):
        pair_dict = {"src": files[i * 2], "trg": files[i * 2 + 1], "dist": -1}
        random_pairs.append(pair_dict)

    return random_pairs


def chamfer_distance(pcd_a, pcd_b):
    """
    Computes the chamfer distance between two sets of points A and B.
    """
    tree = KDTree(pcd_b)
    dist_a = tree.query(pcd_a)[0]
    tree = KDTree(pcd_a)
    dist_b = tree.query(pcd_b)[0]
    return np.mean(dist_a) + np.mean(dist_b)


def npz_generator(object_type: str, viewing_directions: list):
    path = "obj_models/obj_models_small_size_final/"
    names = {}
    directory_path = base_path + path + object_type + "/"
    for filename in os.listdir(directory_path):
        if filename.endswith(".obj"):
            obj_name = filename.split(".")[0]
            point_cloud = obj_to_point_cloud(directory_path + filename)
            point_cloud = scale_shift(point_cloud)
            pts = np.array(point_cloud.points)
            color = np.zeros(pts.shape)
            saving_file = "npz_files/" + object_type
            np.savez(saving_file + obj_name, pcd=pts, color=color)
            for i, viewing_direction in enumerate(viewing_directions):
                pcd = augmenting_the_mesh(directory_path + filename, viewing_direction)
                pts = np.array(pcd.points)
                color = np.zeros(pts.shape)
                saving_file = "npz_files/"
                np.savez(saving_file + obj_name + "00" + str(i), pcd=pts, color=color)


item_types = ["shoe", "teapot", "cutlery"]
viewing_directions = [[0, 5, 1], [2, 2, 0], [-1, -2, 0],
                      [1, 0, 2], [0, 3, 7], [-2, 4, -1],
                      [5, 0, 1], [1, 1, 1], [1, 6, 3]]

names = {"shoe": {}, "teapot": {}, "cutlery": {}}
obj_names = {"shoe": [], "teapot": [], "cutlery": []}
for item in item_types:
    # NPZ Generation
    # npz_generator(object_type=item, viewing_directions=viewing_directions)

    # Splits Creation
    directory_path = base_path + "obj_models/obj_models_small_size_final/" + item + "/"
    for filename in os.listdir(directory_path):
        if filename.endswith(".obj"):
            obj_names[item].append(filename.split(".")[0])

    for name in obj_names[item]:
        pairs = get_random_file_pairs(directory=base_path + "category-level-alignment-3d-geofeatures/mesh_transform"
                                                            "/npz_files/" + item + "/",
                                      obj_name=name,
                                      num_pairs=20)
        names[item][name] = pairs

for item in item_types:
    for name in obj_names[item]:
        name_pairs = names[item][name]
        for pair in name_pairs:
            pcd_a = load_pcd_from_file(npz_name=pair["src"])
            pcd_b = load_pcd_from_file(npz_name=pair["trg"])
            pair["dist"] = chamfer_distance(pcd_a, pcd_b)

# File path to save the pickle file
pickle_file_path = "src_trg_dist.pickle"

# Save the dictionary to a pickle file
# with open(pickle_file_path, 'wb') as f:
#    pickle.dump(names, f)

# print("Dictionary saved to pickle file:", pickle_file_path)
print(names)

for item in item_types:
    for name in obj_names[item]:
        name_pairs = names[item][name]
        filename = f"housecat_6d_test/{name}"
        for pair in name_pairs:
            with open(filename + ".txt", "w") as file:
                file.write(pair["src"] + " " + pair["trg"] + " " + str("{:.3f}".format(pair["dist"])) + "\n")
            if 0.05 < pair["dist"] < 0.1:
                with open(filename + "-0.05.txt", "w") as file:
                    file.write(pair["src"] + " " + pair["trg"] + " " + str("{:.3f}".format(pair["dist"])) + "\n")
            if 0.1 <= pair["dist"] < 0.15:
                with open(filename + "-0.1.txt", "w") as file:
                    file.write(pair["src"] + " " + pair["trg"] + " " + str("{:.3f}".format(pair["dist"])) + "\n")
            if 0.15 <= pair["dist"]:
                with open(filename + "-0.15.txt", "w") as file:
                    file.write(pair["src"] + " " + pair["trg"] + " " + str("{:.3f}".format(pair["dist"])) + "\n")

