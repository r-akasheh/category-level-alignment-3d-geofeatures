import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import os

from mesh_transform import base_path


def transform_depth_rgbd_to_pcd(scene: str = "scene01", image_nr: str = "000001", visualize: bool = False):
    color_raw = o3d.io.read_image(base_path + "scene1-10/{}/rgb/{}.png".format(scene, image_nr))
    depth_raw = o3d.io.read_image(base_path + "scene1-10/{}/depth/{}.png".format(scene, image_nr))

    with open(base_path + "scene1-10/{}/intrinsics.txt".format(scene), "r") as file1:
        f_list = [float(i) for line in file1 for i in line.split(' ') if i.strip()]
    intrinsic_params = np.reshape(np.array(f_list), (3, 3))

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color=color_raw, depth=depth_raw)

    if visualize:
        print(rgbd_image)
        plt.imshow(rgbd_image.color)
        plt.show()

    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.intrinsic_matrix = intrinsic_params

    extrinsic = np.loadtxt(base_path + "/scene1-10/{}/camera_pose/{}.txt".format(scene, image_nr))
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(image=rgbd_image,
                                                         intrinsic=intrinsic,
                                                         extrinsic=extrinsic)

    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    if visualize:
        o3d.visualization.draw_geometries([pcd])
    return pcd, rgbd_image

