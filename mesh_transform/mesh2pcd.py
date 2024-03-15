import open3d as o3d


def obj_to_point_cloud(obj_file: str, number_of_points: int = 20000):
    mesh = o3d.io.read_triangle_mesh(obj_file)
    pc = mesh.sample_points_uniformly(number_of_points=number_of_points)  # Adjust number_of_points as needed
    return pc
