import open3d as o3d
import numpy as np

from mesh_transform import base_path

path_to_npz = base_path + "category-level-alignment-3d-geofeatures/mesh_transform/npz_files/cutlery-fork_1001.npz"

# Load the point cloud data from the npz file
data = np.load(path_to_npz)
points = data['pcd']  # Assuming 'points' is the key for the point cloud data

# Create an Open3D point cloud object
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)

# Visualize the point cloud
o3d.visualization.draw_geometries([point_cloud])
