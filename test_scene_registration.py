import pickle
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

from mesh_transform import base_path
from mesh_transform.transform_depth_rgbd import transform_depth_rgbd_to_pcd

with open(base_path + 'scene1-10/scene07/labels/000007_label.pkl', 'rb') as pickle_file:
    content = pickle.load(pickle_file)

print(content)

scene_pcd, rgbd_image = transform_depth_rgbd_to_pcd(scene="scene07", image_nr="000007", visualize=False)

x1, y1 = content["bboxes"][5][0], content["bboxes"][5][1]
x2, y2 = content["bboxes"][5][2], content["bboxes"][5][3]
print(str(x1), str(y1), str(x2), str(y2))
print(np.asarray(rgbd_image.color))
cropped_color = o3d.geometry.Image(np.asarray(rgbd_image.color, dtype=np.float32)[x1:x2, y1:y2])
cropped_depth = o3d.geometry.Image(np.asarray(rgbd_image.depth, dtype=np.float32)[x1:x2, y1:y2])

with open(base_path + "scene1-10/{}/intrinsics.txt".format("scene07"), "r") as file1:
    f_list = [float(i) for line in file1 for i in line.split(' ') if i.strip()]
intrinsic_params = np.reshape(np.array(f_list), (3, 3))

rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color=cropped_color, depth=cropped_depth)

print(rgbd_image)
plt.imshow(rgbd_image.color)
plt.show()

intrinsic = o3d.camera.PinholeCameraIntrinsic()
intrinsic.intrinsic_matrix = intrinsic_params

extrinsic = np.loadtxt(base_path + "/scene1-10/{}/camera_pose/{}.txt".format("scene07", "000007"))
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(image=rgbd_image,
                                                     intrinsic=intrinsic,
                                                     extrinsic=extrinsic)
o3d.visualization.draw_geometries([pcd])


