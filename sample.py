from mesh_transform import base_path
from mesh_transform.mesh2pcd import obj_to_point_cloud
from mesh_transform.scale_shift import scale_shift

# Provide the path to your 3D mesh file
mesh_file_path = base_path + "obj_models/obj_models_small_size_final/shoe/shoe-skyblue_leda_right.obj"
point_cloud = obj_to_point_cloud(mesh_file_path)
scale_shift(point_cloud, True)