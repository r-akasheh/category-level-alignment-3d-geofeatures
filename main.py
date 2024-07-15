import open3d as o3d
import numpy as np
from spconv.pytorch.utils import PointToVoxel
import cv2
import copy

from mesh_transform.scale_shift import scale_shift, calculate_scaling_variable_and_translate
from utils.ransac_utils import visualize_registration, spconv_vox
from model.lib.utils import to_o3d_pcd, to_o3d_feats, to_array
import spconv.pytorch as spconv
from model.dataset.dataloader import collate_spconv_pair_fn
import torch
from model.resunet_spconv import FCGF_spconv

from mesh_transform import base_path


class ObjectNotFoundError(Exception):
    def __init__(self):
        super().__init__("Object was not found in specified scene. Try another class or scene")

    pass


def registration_prep(scene_pcd, obj_pcd, item_type, voxel_size: float = 0.025, visualize: bool = False,
                      use_baseline: bool = False):
    n_sample_src = len(obj_pcd.points)
    n_sample_trg = len(scene_pcd.points)
    point_cloud_src = scale_shift(obj_pcd)

    scaling_variable, _ = calculate_scaling_variable_and_translate(scene_pcd)
    point_cloud_trg = scene_pcd.scale(1 / (np.sqrt(3) * scaling_variable), center=(0, 0, 0))

    if visualize:
        o3d.visualization.draw_geometries([point_cloud_src, point_cloud_trg])

    src_xyz, tgt_xyz, src_coords, tgt_coords, src_shape, tgt_shape = spconv_vox(np.array(point_cloud_src.points),
                                                                                np.array(point_cloud_trg.points),
                                                                                voxel_size)
    src_features = torch.ones((len(src_coords), 1), dtype=torch.float32)
    tgt_features = torch.ones((len(tgt_coords), 1), dtype=torch.float32)
    list_data = [(src_xyz, tgt_xyz, src_coords, tgt_coords, src_features, tgt_features, torch.ones(1, 2),
                  np.eye(4), src_shape, tgt_shape, None, np.ones((6, 6)))]

    model = FCGF_spconv()

    if use_baseline:
        checkpoint = torch.load("C:/master/robot-vision-modul/FCGF_spconv/checkpoint.pth")
    else:
        checkpoint = torch.load(
            base_path + 'category-level-alignment-3d-geofeatures/model/' + 'snapshot/final_models/' +
            item_type + '/checkpoints/model_best_recall.pth')

    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    model = model.cuda()

    input_dict = collate_spconv_pair_fn(list_data)
    for k, v in input_dict.items():  # load inputs to device.
        if type(v) == list:
            input_dict[k] = [item.cuda() for item in v]
        elif type(v) == torch.Tensor:
            input_dict[k] = v.cuda(0)
        else:
            pass

    src_sp_tensor = spconv.SparseConvTensor(input_dict['src_F'],
                                            input_dict['src_C'].int(),
                                            src_shape, batch_size=1)
    tgt_sp_tensor = spconv.SparseConvTensor(input_dict['tgt_F'],
                                            input_dict['tgt_C'].int(),
                                            tgt_shape, batch_size=1)
    ### get conv features ###
    with torch.no_grad():
        out_src = model(src_sp_tensor)
        out_tgt = model(tgt_sp_tensor)
    src_pcd = input_dict['pcd_src']
    tgt_pcd = input_dict['pcd_tgt']
    src_feats = out_src.features
    tgt_feats = out_tgt.features

    return n_sample_src, n_sample_trg, src_pcd, tgt_pcd, src_feats, tgt_feats, point_cloud_src, point_cloud_trg


def ransac(n_sample_src, n_sample_trg, src_pcd, tgt_pcd,
           src_feats, tgt_feats, point_cloud_src, point_cloud_trg, voxel_size: float = 0.025, visualize: bool = False):
    src_sel = np.random.choice(len(src_pcd), min(len(src_pcd), n_sample_src), replace=False)
    tgt_sel = np.random.choice(len(tgt_pcd), min(len(tgt_pcd), n_sample_trg), replace=False)

    src_pcd = src_pcd[src_sel]
    tgt_pcd = tgt_pcd[tgt_sel]
    src_feats = src_feats[src_sel]
    tgt_feats = tgt_feats[tgt_sel]

    result_ransac = o3d.registration.registration_ransac_based_on_feature_matching(
        to_o3d_pcd(src_pcd), to_o3d_pcd(tgt_pcd), to_o3d_feats(src_feats), to_o3d_feats(tgt_feats),
        voxel_size * 1.5,
        o3d.registration.TransformationEstimationPointToPoint(False), 3,
        [o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.registration.CorrespondenceCheckerBasedOnDistance(voxel_size * 1.5)],
        o3d.registration.RANSACConvergenceCriteria(50000, 1000))
    pred_trans = result_ransac.transformation

    if visualize:
        visualize_registration(point_cloud_src, point_cloud_trg, pred_trans)

    return pred_trans


def extract_obj_pcd_from_scene_o3d(base_path, obj: str = None, scene_nr: str = "01", image_nr: str = "000001",
                                   visualize: bool = False, verbose: bool = False):
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
                                                         extrinsic=extrinsic)
    # extrinsic=np.linalg.inv(extrinsic))
    pcd.transform([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    if visualize:
        o3d.visualization.draw_geometries([pcd], mesh_show_back_face=False)
    if verbose:
        print("Processed scene successfully")
    return pcd, obj_name


def extract_obj_pcd_from_scene(base_path, obj: str = None, scene_nr: str = "01", image_nr: str = "000001",
                               visualize: bool = False):
    file_path = base_path + "scene1-10/scene{}/meta.txt".format(str(scene_nr))
    obj_name = ""
    with open(file_path, 'r') as file:
        for line in file:
            if obj in line:
                # Process each line here
                obj_name = line.strip()[4:]
                instance_id = line.strip()[0]

    if not obj_name:
        raise ObjectNotFoundError

    mask = base_path + "scene1-10/scene{}/instance/{}_{}.png".format(str(scene_nr), str(image_nr), str(obj_name))
    depth = base_path + "scene1-10/scene{}/depth/{}.png".format(str(scene_nr), str(image_nr))

    rgb_image = cv2.imread(mask)
    depth_image = cv2.imread(depth, -1)
    extrinsics = np.loadtxt(base_path + "scene1-10/scene{}/camera_pose/{}.txt".format(scene_nr, image_nr))
    intrinsics = np.loadtxt(base_path + "scene1-10/scene{}/intrinsics.txt".format(scene_nr))

    # Assuming intrinsics matrix is in the form [fx, 0, cx; 0, fy, cy; 0, 0, 1]
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]

    # Create point cloud
    points = []
    colors = []

    height, width = depth_image.shape

    """for v in range(height):
        for u in range(width):
            color = rgb_image[v, u]  # Assuming color image is already normalized
            z = depth_image[v, u] / 1000.0  # Assuming depth image is in millimeters
            if z == 0: continue
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            points.append([x, y, z])
            colors.append(color)
    """

    # Assuming rgb_image, depth_image, cx, cy, fx, fy are defined
    height, width, _ = rgb_image.shape

    # Normalize depth image to avoid division by zero
    normalized_depth_image = depth_image / 1000.0
    valid_depth_mask = normalized_depth_image > 0

    # Create meshgrid for pixel coordinates
    u, v = np.meshgrid(np.arange(width), np.arange(height))

    # Compute x, y, z coordinates for all pixels
    z = normalized_depth_image[valid_depth_mask]
    x = (u[valid_depth_mask] - cx) * z / fx
    y = (v[valid_depth_mask] - cy) * z / fy

    # Extract colors for valid depth pixels
    colors = rgb_image[valid_depth_mask]

    # Stack the coordinates to form the points array
    points = np.column_stack((x, y, z))

    # Convert colors to list if needed (optional, depending on use case)
    colors = colors.tolist()

    # Convert points to list if needed (optional, depending on use case)
    points = points.tolist()

    # Convert to Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors) / 255.0)

    # Save point cloud
    # o3d.io.write_point_cloud('pointcloud.ply', pcd)
    ttest = (np.array(pcd.points))[~np.all(np.array(pcd.colors) == 1, axis=1)]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(ttest)
    if visualize:
        o3d.visualization.draw_geometries([pcd])

    return copy.deepcopy(pcd), obj_name, (int(instance_id) - 1)


def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    return inlier_cloud


def retrieve_obj_pcd(obj_name):
    path_to_npz = base_path + "category-level-alignment-3d-geofeatures/model/dataset/housecat_6d/FCGF_data/housecat_6d/" + obj_name + ".npz"

    # Load the point cloud data from the npz file
    data = np.load(path_to_npz)
    points = data['pcd']  # Assuming 'points' is the key for the point cloud data

    # Create an Open3D point cloud object
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    return point_cloud
