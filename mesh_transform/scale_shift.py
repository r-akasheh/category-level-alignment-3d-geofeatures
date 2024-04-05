import open3d as o3d
import numpy as np


def calculate_scaling_variable_and_translate(point_cloud):
    # Calculate mean of points across each axis
    test = np.array(point_cloud.points)
    x_translation = - np.mean(test[:, 0])
    y_translation = - np.mean(test[:, 1])
    z_translation = - np.mean(test[:, 2])

    # Translate point cloud by calculated means
    translated_pc = point_cloud.translate((x_translation, y_translation, z_translation))

    # Create a np array from the newly translate point cloud
    pc_array = np.array(translated_pc.points)

    # Calculate the largest range among the three axes
    diff_x = np.max(pc_array[:, 0]) - np.min(pc_array[:, 0])
    diff_y = np.max(pc_array[:, 1]) - np.min(pc_array[:, 1])
    diff_z = np.max(pc_array[:, 2]) - np.min(pc_array[:, 2])

    # Calculate the scaling variable according to the ranges
    scaling_variable = np.max((diff_x, diff_y, diff_z))

    return scaling_variable, translated_pc


def get_lines() -> list:
    lines = [
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 3],
        [4, 5],
        [4, 6],
        [5, 7],
        [6, 7],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]
    return lines


def get_line_set(points):
    lines = get_lines()
    colors = [[1, 0, 0] for i in range(len(lines))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def scale_shift(point_cloud: o3d.geometry.PointCloud, visualize: bool = False) -> o3d.geometry.PointCloud:
    # Visualize the point cloud
    if visualize:
        o3d.visualization.draw_geometries([point_cloud], mesh_show_back_face=False)

    scaling_variable, translated_pc = calculate_scaling_variable_and_translate(point_cloud=point_cloud)

    # Create a new point cloud from the last one by sqrt(3)/scaling_variable, to achieve a unit diagonal
    translated_pc = translated_pc.scale(1 / (np.sqrt(3) * scaling_variable), center=(0, 0, 0))

    # Create a coordinate frame with origin 0,0,0
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.1, origin=[0, 0, 0])

    # Create a bounding box with unit diagonal. Each side of a cube with a unit diagonal is equal to 1/np.sqrt(3).
    # To get the coordinates of each of the 8 corners of a bounding box, we have to divide that length by 2
    if visualize:
        scaler = (1 / np.sqrt(3)) / 2
        points = np.array([
            [-scaler, -scaler, -scaler],
            [scaler, -scaler, -scaler],
            [-scaler, scaler, -scaler],
            [scaler, scaler, -scaler],
            [-scaler, -scaler, scaler],
            [scaler, -scaler, scaler],
            [-scaler, scaler, scaler],
            [scaler, scaler, scaler],
        ])

        line_set = get_line_set(points)
        o3d.visualization.draw_geometries([translated_pc, mesh_frame, line_set], mesh_show_back_face=False)

    return translated_pc


def scale_shift_nocs_color_demonstration(point_cloud: o3d.geometry.PointCloud, five_positions: np.array,
                              visualize: bool = False) -> o3d.geometry.PointCloud:
    # Visualize the point cloud
    if visualize:
        o3d.visualization.draw_geometries([point_cloud], mesh_show_back_face=False)

    scaling_variable, translated_pc = calculate_scaling_variable_and_translate(point_cloud)

    # Create a new point cloud from the last one by /scaling_variable, to achieve a unit length
    translated_pc = translated_pc.scale(1 / scaling_variable, center=(0, 0, 0))
    translated_pc = translated_pc.translate((0.5, 0.5, 0.5))

    colors = []
    for point in translated_pc.points:
        x, y, z = point
        r = x  # Red component (grows from 0 to 1 along x-axis)
        g = y  # Green component (grows from 0 to 1 along y-axis)
        b = z  # Blue component (grows from 0 to 1 along z-axis)
        colors.append([r, g, b])
    translated_pc.colors = o3d.utility.Vector3dVector(colors)

    spheres = []

    for position in range(0, five_positions.shape[0]):
        mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=.025)
        mesh_sphere.compute_vertex_normals()
        mesh_sphere.paint_uniform_color([0.9, 0.1, 0.1])
        mesh_sphere.translate(five_positions[position])
        spheres.append(mesh_sphere)

    # Create a coordinate frame with origin 0,0,0
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.1, origin=[0, 0, 0])

    # Create a bounding box with unit diagonal. Each side of a cube with a unit diagonal is equal to 1/np.sqrt(3).
    # To get the coordinates of each of the 8 corners of a bounding box, we have to divide that length by 2
    if visualize:
        points = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ])

        line_set = get_line_set(points)

        o3d.visualization.draw_geometries([translated_pc, mesh_frame, line_set] + spheres, mesh_show_back_face=False)

    return translated_pc
