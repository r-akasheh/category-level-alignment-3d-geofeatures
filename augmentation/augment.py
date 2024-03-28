import numpy as np
import open3d as o3d
import pyrender
import trimesh


def augmenting_the_pcd(obj_file: str, view_direction: np.array, number_of_points: int = 20000):
    mesh = o3d.io.read_triangle_mesh(obj_file)
    mesh.compute_vertex_normals()
    front_triangles = []
    for i in range(len(mesh.triangles)):
        normal = mesh.triangle_normals[i]
        dot_product = normal.dot(view_direction)
        if dot_product > 0.5:
            front_triangles.append(mesh.triangles[i])

    front_mesh = o3d.geometry.TriangleMesh()
    front_mesh.vertices = mesh.vertices
    front_mesh.triangles = o3d.utility.Vector3iVector(front_triangles)
    pcd = front_mesh.sample_points_uniformly(number_of_points=number_of_points)  # Adjust number_of_points as needed
    return pcd


def augmenting_from_depth(obj_file: str, visualize: bool = False,
                          s=np.sqrt(2) / 2):  # s is the parameter for the orientation, camera pose should be improved
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

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.cpu.pybind.geometry.Image(color),
                                                                    o3d.cpu.pybind.geometry.Image(depth))

    intrinsic = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image,
                                                         intrinsic)
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    if visualize:
        o3d.visualization.draw_geometries([pcd])

    return pcd
