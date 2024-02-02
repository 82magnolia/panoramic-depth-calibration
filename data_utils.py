import numpy as np
import json
from typing import Tuple, Union
from pandas import read_table
from scipy.spatial.transform import Rotation
from glob import glob


# Stanford

def read_stanford(filepath: str, sample_rate: float = 1) -> Tuple[np.array, np.array]:
    """
    Read Stanford2D-3D-S point cloud data from filepath

    Args:
        filepath: full path name
        sample_rate: point cloud sampling rate (at least 1), returns point cloud of size (N // sample_rate, 3)

    Returns:
        xyz: (N, 3) numpy array containing xyz coordinates of the point cloud data
        rgb: (N, 3) numpy array containing rgb values of the point cloud data, in range of [0, 1]
    """

    # read file
    data = read_table(filepath, header=None, delim_whitespace=True).values

    xyz = data[:, :3]
    rgb = data[:, 3:] / 255.

    # sampling point cloud
    if sample_rate > 1.0:
        perm = np.random.permutation(xyz.shape[0])
        num_samples = int(xyz.shape[0] / sample_rate)
        idx = perm[:num_samples]
        xyz = xyz[idx]
        rgb = rgb[idx]

    return xyz, rgb


def obtain_gt_stanford(area_num: Union[int, str], img_name: str, random_rot: bool = False) -> Tuple[np.array, np.array]:
    """
    Obtain Stanford2D-3D-S dataset ground truth translation & rotation

    Args:
        area_num: area number of the data
        img_name: panorama image name
        random_rot: if True, applies random rotation to ground truth 
    
    Returns:
        gt_trans: (3, 1) numpy array containing ground truth translation
        gt_rot: (3, 3) numpy array containing ground truth rotation
    """

    if area_num < 10:
        splits = img_name.split('_')
        camera_id = splits[1]
        room_type = splits[2]
        room_id = splits[3]

        pose_file = './data/stanford/pose/area_{}/camera_{}_{}_{}_frame_equirectangular_domain_pose.json'.format(area_num, camera_id, room_type, room_id)

        with open(pose_file) as f:
            pose_file = json.load(f)
        
        cam_loc = np.array(pose_file['camera_location'])
        cam_rot = pose_file['final_camera_rotation']

        # ground truth translation
        trans = [[cam_loc[0]], [cam_loc[1]], [cam_loc[2]]]
        gt_trans = np.array(trans)

        # ground truth rotation
        r = Rotation.from_euler('xyz', cam_rot)
        r = r.as_matrix()
        rot = np.zeros([3, 3])

        rot[:, 0] = r[:, 2]
        rot[:, 1] = r[:, 0]
        rot[:, 2] = r[:, 1]

        rot = np.linalg.inv(rot)

        # gt_rot is always 180 degrees rotated along z-axis
        flip_mat = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        gt_rot = np.matmul(flip_mat, rot)

    else:
        splits = img_name.split('_')
        camera_id = splits[1]
        room_type = splits[2]
        room_id = splits[3]

        transformation_file = './data/stanford/pose/area_{}/{}_{}.txt'.format(area_num, room_type, room_id)
        pose_file = './data/stanford/pose/area_{}/camera_{}_{}_{}_frame_equirectangular_domain_pose.json'.format(area_num // 10, camera_id, room_type, room_id)

        with open(pose_file) as f:
            pose_file = json.load(f)
        
        cam_loc = np.array(pose_file['camera_location'])
        cam_rot = pose_file['final_camera_rotation']

        # ground truth translation
        trans = [[cam_loc[0]], [cam_loc[1]], [cam_loc[2]]]
        gt_trans = np.array(trans)

        # ground truth rotation
        r = Rotation.from_euler('xyz', cam_rot)
        r = r.as_matrix()
        rot = np.zeros([3, 3])

        rot[:, 0] = r[:, 2]
        rot[:, 1] = r[:, 0]
        rot[:, 2] = r[:, 1]

        rot = np.linalg.inv(rot)
        """
        # gt_rot is always 180 degrees rotated along z-axis
        flip_mat = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        gt_rot = np.matmul(flip_mat, rot)
        """
        transformation_mat = np.loadtxt(transformation_file)
        rot_mat = transformation_mat[:, :3]
        trans_mat = transformation_mat[:, 3:]

        gt_rot = np.matmul(rot, np.linalg.inv(rot_mat))
        flip_mat = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        gt_rot = np.matmul(flip_mat, gt_rot)
        gt_trans = np.matmul(rot_mat, gt_trans - trans_mat)

    return gt_trans, gt_rot


def read_omniscenes(filepath: str, sample_rate: float = 1) -> Tuple[np.array, np.array]:
    """
    Read omniscenes dataset point cloud data from filepath

    Args:
        filepath: full path name
        sample_rate: point cloud sampling rate (at least 1)

    Returns:
        xyz: (N / sample_rate, 3) numpy array containing xyz coordinates of the point cloud data
        rgb: (N / sample_rate, 3) numpy array containing rgb values of the point cloud data, in range of [0, 1]
    """

    # read file
    data = read_table(filepath, header=None, delim_whitespace=True).values

    xyz = data[:, :3]
    rgb = data[:, 3:] / 255.

    # sampleing point cloud
    if sample_rate > 1.0:
        perm = np.random.permutation(xyz.shape[0])
        num_samples = int(xyz.shape[0] / sample_rate)
        idx = perm[:num_samples]
        xyz = xyz[idx]
        rgb = rgb[idx]

    return xyz, rgb


def obtain_gt_omniscenes(full_img_path: str) -> Tuple[np.array, np.array]:
    """
    Obtain omniscenes dataset ground truth translation & rotation

    Args:
        img_name: panorama image name
    
    Returns:
        gt_trans: (3, 1) numpy array containing gound truth translation
        gt_rot: (3, 3) numpy array containing ground truth rotation
    """

    pose_file = full_img_path.replace('pano', 'pose').replace('.jpg', '.txt')
    gt_mat = np.loadtxt(pose_file)
    gt_rot = gt_mat[:, :3]
    gt_trans = gt_mat[:, 3:]

    return gt_trans, gt_rot


# Wrapping functions for cleaner loading

def get_filename(dataset, **kwargs):
    if dataset == 'omniscenes':
        filenames = sorted(glob(f"./data/omniscenes/{kwargs['split_type']}_pano/*/*"))
    elif dataset == 'stanford':
        if kwargs['split_type'] is None:
            filenames = sorted(glob(f"./data/stanford/pano/*/*.png"))
            filenames = sorted(filenames, key=lambda x: f"{x.split('/')[-2]}_{x.split('/')[-1].split('_')[-6]}_{x.split('/')[-1].split('_')[-5]}")
        else:
            filenames = sorted(glob(f"./data/stanford/pano/{kwargs['split_type']}/*.png"))
            filenames = sorted(filenames, key=lambda x: f"{x.split('/')[-2]}_{x.split('/')[-1].split('_')[-6]}_{x.split('/')[-1].split('_')[-5]}")
    elif dataset == 'matterport':
        filenames = sorted(glob(f"./data/matterport_depth/**/*_rgb.png"))
    if kwargs.get('room_type', None) is not None:
        if kwargs['room_type'] == 'small':
            filenames = [f for f in filenames if 'wedding' not in f]        
        else:
            filenames = [f for f in filenames if kwargs['room_type'] in f]

    return filenames


def get_pcd_name(dataset, **kwargs):
    if dataset == 'omniscenes':
        pcd_name = "./data/omniscenes/pcd/{}_{}.txt".format(kwargs['room_type'], kwargs['room_no'])
    elif dataset == 'stanford':
        pcd_name = "./data/stanford/pcd/{}/{}_{}.txt".format(kwargs['area_name'], kwargs['room_type'], kwargs['room_no'])

    return pcd_name


def read_pcd(dataset, **kwargs):
    if dataset == 'omniscenes':
        return read_omniscenes(kwargs['pcd_name'], sample_rate=kwargs['sample_rate'])
    elif dataset == 'stanford':
        return read_stanford(kwargs['pcd_name'], sample_rate=kwargs['sample_rate'])


def read_gt(dataset, **kwargs):
    if dataset == 'omniscenes':
        return obtain_gt_omniscenes(kwargs['filename'])
    elif dataset == 'stanford':
        return obtain_gt_stanford(kwargs['area_num'], kwargs['img_name'])
