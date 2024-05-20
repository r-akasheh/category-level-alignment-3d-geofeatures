import open3d as o3d
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from mesh_transform import base_path


class ObjectNotFoundError(Exception):
    def __init__(self):
        super().__init__("Object was not found in specified scene. Try another class or scene")

    pass


def extract_obj_pcd_from_scene(base_path, obj: str = None, scene_nr: str = "01", image_nr: str = "000001"):
    scene_path = base_path + "scene1-10/scene{}/rgb/{}.png".format(str(scene_nr), str(image_nr))
    file_path = base_path + "scene1-10/scene{}/meta.txt".format(str(scene_nr))
    obj_name = ""
    with open(file_path, 'r') as file:
        for line in file:
            if obj in line:
                # Process each line here
                obj_name = line.strip()[4:]

    if not obj_name:
        raise ObjectNotFoundError

    mask = base_path + "scene1-10/scene{}/instance/{}_{}.png".format(str(scene_nr), str(image_nr), str(obj_name))
    depth = base_path + "scene1-10/scene{}/depth/{}.png".format(str(scene_nr), str(image_nr))

    color_raw = np.array(o3d.io.read_image(scene_path))
    depth_raw = np.array(o3d.io.read_image(depth))
    mask_raw = ~np.array(o3d.io.read_image(mask))
    color_raw[mask_raw == 0] = 0
    depth_raw[mask_raw == 0] = 0

    height = color_raw.shape[0]
    width = color_raw.shape[1]

    with open(base_path + "scene1-10/scene{}/intrinsics.txt".format(scene_nr), "r") as file1:
        f_list = [float(i) for line in file1 for i in line.split(' ') if i.strip()]

    intrinsic_params = np.reshape(np.array(f_list), (3, 3))
    color_raw = o3d.geometry.Image(color_raw.astype(np.uint8))
    depth_raw = o3d.geometry.Image(depth_raw.astype(np.uint8))

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color=color_raw, depth=depth_raw)

    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.intrinsic_matrix = intrinsic_params
    intrinsic.width = width
    intrinsic.height = height
    extrinsic = np.loadtxt(base_path + "scene1-10/scene{}/camera_pose/{}.txt".format(scene_nr, image_nr))
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(image=rgbd_image,
                                                         intrinsic=intrinsic,
                                                         extrinsic=np.linalg.inv(extrinsic))
    pcd.transform([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    o3d.visualization.draw_geometries([pcd], mesh_show_back_face=False)
    print("Processed scene successfully")


extract_obj_pcd_from_scene(base_path, obj="shoe", image_nr="000003", scene_nr="01")
