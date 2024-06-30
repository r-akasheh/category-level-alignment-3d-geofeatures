import numpy as np
import open3d as o3d
import pyrender
import trimesh

from mesh_transform.scale_shift import scale_shift


def augmenting_the_mesh(obj_file: str, view_direction: np.array, number_of_points: int = 2000,
                        visualize: bool = False):
    mesh = o3d.io.read_triangle_mesh(obj_file)
    mesh.compute_vertex_normals()
    front_triangles = []
    for i in range(len(mesh.triangles)):
        normal = mesh.triangle_normals[i]
        view_direction_norm = view_direction / np.linalg.norm(view_direction)
        dot_product = normal.dot(view_direction_norm)

        if dot_product > 0.5:
            front_triangles.append(mesh.triangles[i])

    front_mesh = o3d.geometry.TriangleMesh()
    front_mesh.vertices = mesh.vertices
    front_mesh.triangles = o3d.utility.Vector3iVector(front_triangles)
    pcd = front_mesh.sample_points_uniformly(number_of_points=number_of_points)  # Adjust number_of_points as needed
    pcd = scale_shift(pcd)
    if visualize:
        o3d.visualization.draw_geometries([pcd])

    return pcd


def augmenting_from_depth(obj_file: str, visualize: bool = False,
                          s=np.sqrt(2) / 2):
    # TODO s is the parameter for the orientation; camera pose should be improved
    mesh = trimesh.load(obj_file)
    mesh = pyrender.Mesh.from_trimesh(mesh)
    scene = pyrender.Scene()
    scene.add(mesh)
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)

    camera_pose = np.array([
        [0.0, -s, s, 0.3],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, s, s, 0.35],
        [0.0, 0.0, 0.0, 1.0],
    ])
    scene.add(camera, pose=camera_pose)
    light = pyrender.SpotLight(color=np.ones(3), intensity=3.0,
                               innerConeAngle=np.pi / 16.0,
                               outerConeAngle=np.pi / 6.0)
    scene.add(light, pose=camera_pose)
    r = pyrender.OffscreenRenderer(400, 400)

    color, depth = r.render(scene)

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color=o3d.geometry.Image(color.astype(np.uint8)),
                                                                    depth=o3d.geometry.Image(depth))
    intrinsic = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)

    pcd.transform(camera_pose)
    pcd.transform([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    pcd = scale_shift(pcd)
    if visualize:
        o3d.visualization.draw_geometries([pcd])

    return pcd


def random_remove_points_with_spared_radius(point_cloud, removal_fraction, spared_radius,
                                            center_point=None, visualize: bool = False):
    # Convert Open3D point cloud to NumPy array
    points = np.asarray(point_cloud.points)
    num_points = len(points)

    # Determine the center point (use origin if none is provided)
    if center_point is None:
        center_point = np.mean(points, axis=0)  # Center of the point cloud

    # Calculate distances from the center point
    distances = np.linalg.norm(points - center_point, axis=1)

    # Create a mask for points outside the spared radius
    mask_spared = distances > spared_radius

    # Points outside the spared radius
    points_outside_spared = points[mask_spared]

    # Determine the number of points to remove from outside the spared radius
    num_remove = int(removal_fraction * len(points_outside_spared))

    # Randomly select indices to remove
    remove_indices = np.random.choice(len(points_outside_spared), num_remove, replace=False)

    # Create mask to filter out the points
    mask_remove = np.ones(len(points_outside_spared), dtype=bool)
    mask_remove[remove_indices] = False

    # Combine spared points and remaining points outside the spared radius
    remaining_points = points_outside_spared[mask_remove]
    spared_points = points[~mask_spared]
    new_points = np.concatenate((remaining_points, spared_points), axis=0)

    # Convert back to Open3D point cloud
    new_point_cloud = o3d.geometry.PointCloud()
    new_point_cloud.points = o3d.utility.Vector3dVector(new_points)

    if visualize:
        o3d.visualization.draw_geometries([new_point_cloud])
    return new_point_cloud


def extract_bounding_box_region(point_cloud, center, size, visualize: bool = False):
    """
    Extract a region of the point cloud defined by a bounding box.

    :param visualize:
    :param point_cloud: Input point cloud (Open3D point cloud object).
    :param center: Center of the bounding box (numpy array of shape (3,)).
    :param size: Size of the bounding box (numpy array of shape (3,) or float).
    :return: Extracted region as an Open3D point cloud.
    """
    # Convert to NumPy array
    points = np.asarray(point_cloud.points)

    # Create bounding box
    if isinstance(size, (int, float)):
        size = np.array([size, size, size])
    min_bound = center - size / 2
    max_bound = center + size / 2

    # Create mask for points inside the bounding box
    mask = np.all((points >= min_bound) & (points <= max_bound), axis=1)

    # Extract points within the bounding box
    extracted_points = points[mask]

    # Convert back to Open3D point cloud
    extracted_point_cloud = o3d.geometry.PointCloud()
    extracted_point_cloud.points = o3d.utility.Vector3dVector(extracted_points)

    if visualize:
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([extracted_point_cloud, coordinate_frame])

    return extracted_point_cloud


def extract_spherical_region(point_cloud, center, radius, visualize: bool = False):
    """
    Extract a region of the point cloud defined by a sphere.

    :param point_cloud: Input point cloud (Open3D point cloud object).
    :param center: Center of the sphere (numpy array of shape (3,)).
    :param radius: Radius of the sphere.
    :return: Extracted region as an Open3D point cloud.
    """
    # Convert to NumPy array
    points = np.asarray(point_cloud.points)

    # Calculate distances from the center
    distances = np.linalg.norm(points - center, axis=1)

    # Create mask for points inside the sphere
    mask = distances <= radius

    # Extract points within the sphere
    extracted_points = points[mask]

    # Convert back to Open3D point cloud
    extracted_point_cloud = o3d.geometry.PointCloud()
    extracted_point_cloud.points = o3d.utility.Vector3dVector(extracted_points)

    if visualize:
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([extracted_point_cloud, coordinate_frame])

    return extracted_point_cloud
