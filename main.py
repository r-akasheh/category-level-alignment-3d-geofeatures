import open3d as o3d
import numpy as np
from spconv.pytorch.utils import PointToVoxel

from mesh_transform.scale_shift import scale_shift, calculate_scaling_variable_and_translate
from utils.ransac_utils import visualize_registration
from model.lib.utils import to_o3d_pcd, to_o3d_feats, to_array
import spconv.pytorch as spconv
from model.dataset.dataloader import collate_spconv_pair_fn
import torch
from model.resunet_spconv import FCGF_spconv

from mesh_transform import base_path
from utils.ransac_utils import spconv_vox


class ObjectNotFoundError(Exception):
    def __init__(self):
        super().__init__("Object was not found in specified scene. Try another class or scene")

    pass


def load_features_run_ransac(scene_pcd, obj_pcd, item_type):
    voxel_size = 0.025
    n_sample_src = 20000
    n_sample_trg = len(scene_pcd.points)
    point_cloud_src = scale_shift(obj_pcd)

    scaling_variable, _ = calculate_scaling_variable_and_translate(scene_pcd)
    point_cloud_trg = scene_pcd.scale(1 / (np.sqrt(3) * scaling_variable), center=(0, 0, 0))

    o3d.visualization.draw_geometries([point_cloud_src, point_cloud_trg])

    src_xyz, tgt_xyz, src_coords, tgt_coords, src_shape, tgt_shape = spconv_vox(np.array(point_cloud_src.points),
                                                                                np.array(point_cloud_trg.points),
                                                                                voxel_size)
    src_features = torch.ones((len(src_coords), 1), dtype=torch.float32)
    tgt_features = torch.ones((len(tgt_coords), 1), dtype=torch.float32)
    list_data = [(src_xyz, tgt_xyz, src_coords, tgt_coords, src_features, tgt_features, torch.ones(1, 2),
                  np.eye(4), src_shape, tgt_shape, None, np.ones((6, 6)))]

    ## init model
    model = FCGF_spconv()
    checkpoint = torch.load(base_path + 'category-level-alignment-3d-geofeatures/model/' + 'snapshot/final_models/' + item_type + '/checkpoint.pth')
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

    ## register
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

    visualize_registration(point_cloud_src, point_cloud_trg, pred_trans)


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
    return pcd, obj_name


def retrieve_obj_pcd(obj_name):
    path_to_npz = base_path + "category-level-alignment-3d-geofeatures/model/dataset/housecat_6d/FCGF_data/housecat_6d/" + obj_name + ".npz"

    # Load the point cloud data from the npz file
    data = np.load(path_to_npz)
    points = data['pcd']  # Assuming 'points' is the key for the point cloud data

    # Create an Open3D point cloud object
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    return point_cloud


#scene_pcd, obj_name = extract_obj_pcd_from_scene(base_path, obj="cutlery", image_nr="000003", scene_nr="02")
#scene_pcd, obj_name = extract_obj_pcd_from_scene(base_path, obj="cutlery", image_nr="000003", scene_nr="03")
scene_pcd, obj_name = extract_obj_pcd_from_scene(base_path, obj="shoe", image_nr="000113", scene_nr="07")
print(obj_name)
item_type = obj_name.split("-")[0]
obj_pcd = retrieve_obj_pcd(obj_name)
o3d.io.write_point_cloud(filename="./src_pcd.ply", pointcloud=scene_pcd)
o3d.io.write_point_cloud(filename="./trg_pcd.ply", pointcloud=obj_pcd)
load_features_run_ransac(scene_pcd, obj_pcd, item_type)
