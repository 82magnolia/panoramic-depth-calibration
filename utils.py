import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, Union
import os
import random
from math import sin, cos, ceil, floor
import math
from skimage.util import random_noise


def cloud2idx(xyz: torch.Tensor, batched: bool = False) -> torch.Tensor:
    """
    Change 3d coordinates to image coordinates ranged in [-1, 1].

    Args:
        xyz: (N, 3) torch tensor containing xyz values of the point cloud data
        batched: If True, performs batched operation with xyz considered as shape (B, N, 3)

    Returns:
        coord_arr: (N, 2) torch tensor containing transformed image coordinates
    """
    if batched:
        # first project 3d coordinates to a unit sphere and obtain vertical/horizontal angle

        # vertical angle
        theta = torch.unsqueeze(torch.atan2((torch.norm(xyz[..., :2], dim=-1)), xyz[..., 2] + 1e-6), -1)  # (B, N, 1)

        # horizontal angle
        phi = torch.atan2(xyz[..., 1:2], xyz[..., 0:1] + 1e-6)  # (B, N, 1)
        phi += np.pi

        sphere_cloud_arr = torch.cat([phi, theta], dim=-1)  # (B, N, 2)

        # image coordinates ranged in [0, 1]
        coord_arr = torch.stack([1.0 - sphere_cloud_arr[..., 0] / (np.pi * 2), sphere_cloud_arr[..., 1] / np.pi], dim=-1)
        # Rearrange so that the range is in [-1, 1]
        coord_arr = (2 * coord_arr - 1)  # (B, N, 2)

    else:
        # first project 3d coordinates to a unit sphere and obtain vertical/horizontal angle

        # vertical angle
        theta = torch.unsqueeze(torch.atan2((torch.norm(xyz[:, :2], dim=-1)), xyz[:, 2] + 1e-6), 1)

        # horizontal angle
        phi = torch.atan2(xyz[:, 1:2], xyz[:, 0:1] + 1e-6)
        phi += np.pi

        sphere_cloud_arr = torch.cat([phi, theta], dim=-1)

        # image coordinates ranged in [0, 1]
        coord_arr = torch.stack([1.0 - sphere_cloud_arr[:, 0] / (np.pi * 2), sphere_cloud_arr[:, 1] / np.pi], dim=-1)
        # Rearrange so that the range is in [-1, 1]
        coord_arr = (2 * coord_arr - 1)

    return coord_arr


def inv_cloud2idx(coord_arr: torch.Tensor):
    # Inversion of cloud2idx: given a (N, 2) coord_arr, returns a set of (N, 3) 3D points on a sphere.
    sphere_cloud_arr = (coord_arr + 1.) / 2.
    sphere_cloud_arr[:, 0] = (1.0 - sphere_cloud_arr[:, 0]) * (2 * np.pi)
    sphere_cloud_arr[:, 1] = np.pi * sphere_cloud_arr[:, 1]  # Contains [phi, theta] of sphere

    sphere_cloud_arr[:, 0] -= np.pi  # Subtraction to accomodate for cloud2idx

    sphere_xyz = torch.zeros(sphere_cloud_arr.shape[0], 3, device=coord_arr.device)
    sphere_xyz[:, 0] = torch.sin(sphere_cloud_arr[:, 1]) * torch.cos(sphere_cloud_arr[:, 0])
    sphere_xyz[:, 1] = torch.sin(sphere_cloud_arr[:, 1]) * torch.sin(sphere_cloud_arr[:, 0])
    sphere_xyz[:, 2] = torch.cos(sphere_cloud_arr[:, 1])

    return sphere_xyz


def sample_from_img(img: torch.Tensor, coord_arr: torch.Tensor, padding='zeros', mode='bilinear', batched=False) -> torch.Tensor:
    """
    Image sampling function
    Use coord_arr as a grid for sampling from img

    Args:
        img: (H, W, 3) torch tensor containing image RGB values
        coord_arr: (N, 2) torch tensor containing image coordinates, ranged in [-1, 1], converted from 3d coordinates
        padding: Padding mode to use for grid_sample
        mode: How to sample from grid
        batched: If True, assumes an additional batch dimension for coord_arr

    Returns:
        sample_rgb: (N, 3) torch tensor containing sampled RGB values
    """
    if batched:
        img = img.permute(2, 0, 1)
        img = torch.unsqueeze(img, 0)

        # sampling from img
        sample_arr = coord_arr.reshape(coord_arr.shape[0], coord_arr.shape[1], 1, 2)
        sample_arr = torch.clip(sample_arr, min=-0.99, max=0.99)
        sample_rgb = F.grid_sample(img.expand(coord_arr.shape[0], -1, -1, -1), sample_arr, mode=mode, align_corners=False, padding_mode=padding)

        sample_rgb = torch.squeeze(sample_rgb)  # (B, 3, N)
        sample_rgb = torch.transpose(sample_rgb, 1, 2)  # (B, N, 3)  

    else:
        img = img.permute(2, 0, 1)
        img = torch.unsqueeze(img, 0)

        # sampling from img
        sample_arr = coord_arr.reshape(1, -1, 1, 2)
        sample_arr = torch.clip(sample_arr, min=-0.99, max=0.99)
        sample_rgb = F.grid_sample(img, sample_arr, mode=mode, align_corners=False, padding_mode=padding)

        sample_rgb = torch.squeeze(torch.squeeze(sample_rgb, 0), 2)
        sample_rgb = torch.transpose(sample_rgb, 0, 1)

    return sample_rgb


def warp_from_img(img: torch.Tensor, coord_arr: torch.Tensor, padding='zeros', mode='bilinear') -> torch.Tensor:
    """
    Image warping function
    Use coord_arr as a grid for warping from img

    Args:
        img: (H, W, C) torch tensor containing image RGB values
        coord_arr: (H, W, 2) torch tensor containing image coordinates, ranged in [-1, 1], converted from 3d coordinates
        padding: Padding mode to use for grid_sample
        mode: How to sample from grid

    Returns:
        sample_rgb: (H, W, C) torch tensor containing sampled RGB values
    """

    img = img.permute(2, 0, 1)  # (C, H, W)
    img = torch.unsqueeze(img, 0)  # (1, C, H, W)

    # sampling from img
    sample_arr = coord_arr.unsqueeze(0)  # (1, H, W, 2)
    sample_arr = torch.clip(sample_arr, min=-0.99, max=0.99)
    sample_rgb = F.grid_sample(img, sample_arr, align_corners=False, padding_mode=padding, mode=mode)  # (1, C, H, W)

    sample_rgb = sample_rgb.squeeze(0).permute(1, 2, 0)  # (H, W, C)

    return sample_rgb


def ij2coord(ij_values, resolution):
    # Convert (N, 2) image ij-coordinates to 3D spherical coordinates
    coord_idx = torch.flip(ij_values.float(), [-1])
    coord_idx[:, 0] /= (resolution[1] - 1)
    coord_idx[:, 1] /= (resolution[0] - 1)

    coord_idx = 2. * coord_idx - 1.

    sphere_xyz = inv_cloud2idx(coord_idx)  # Points on sphere
    return sphere_xyz


def make_pano(xyz: torch.Tensor, rgb: torch.Tensor, resolution: Tuple[int, int] = (200, 400), 
        return_torch: bool = False, return_coord: bool = False, return_norm_coord: bool = False) -> Union[torch.Tensor, np.array]:
    """
    Make panorama image from xyz and rgb tensors

    Args:
        xyz: (N, 3) torch tensor containing xyz coordinates
        rgb: (N, 3) torch tensor containing rgb values, ranged in [0, 1]
        resolution: Tuple size of 2, returning panorama image of size resolution
        return_torch: if True, return image as torch.Tensor
                      if False, return image as numpy.array
        return_coord: If True, return coordinate in long format
        return_norm_coord: If True, return coordinate in normalized float format

    Returns:
        image: (H, W, 3) torch.Tensor or numpy.array
    """

    with torch.no_grad():

        # project farther points first
        dist = torch.norm(xyz, dim=-1)
        mod_idx = torch.argsort(dist)
        mod_idx = torch.flip(mod_idx, dims=[0])
        mod_xyz = xyz.clone().detach()[mod_idx]
        mod_rgb = rgb.clone().detach()[mod_idx]

        orig_coord_idx = cloud2idx(mod_xyz)
        coord_idx = (orig_coord_idx + 1.0) / 2.0
        # coord_idx[:, 0] is x coordinate, coord_idx[:, 1] is y coordinate
        coord_idx[:, 0] *= (resolution[1] - 1)
        coord_idx[:, 1] *= (resolution[0] - 1)

        coord_idx = torch.flip(coord_idx, [-1])
        coord_idx = coord_idx.long()
        save_coord_idx = coord_idx.clone().detach()
        coord_idx = tuple(coord_idx.t())

        image = torch.zeros([resolution[0], resolution[1], 3], dtype=torch.float, device=xyz.device)

        # color the image
        # pad by 1
        temp = torch.ones_like(coord_idx[0], device=xyz.device)
        coord_idx1 = (torch.clamp(coord_idx[0] + temp, max=resolution[0] - 1),
                      torch.clamp(coord_idx[1] + temp, max=resolution[1] - 1))
        coord_idx2 = (torch.clamp(coord_idx[0] + temp, max=resolution[0] - 1),
                      coord_idx[1])
        coord_idx3 = (torch.clamp(coord_idx[0] + temp, max=resolution[0] - 1),
                      torch.clamp(coord_idx[1] - temp, min=0))
        coord_idx4 = (torch.clamp(coord_idx[0] - temp, min=0),
                      torch.clamp(coord_idx[1] + temp, max=resolution[1] - 1))
        coord_idx5 = (torch.clamp(coord_idx[0] - temp, min=0),
                      coord_idx[1])
        coord_idx6 = (torch.clamp(coord_idx[0] - temp, min=0),
                      torch.clamp(coord_idx[1] - temp, min=0))
        coord_idx7 = (coord_idx[0],
                      torch.clamp(coord_idx[1] + temp, max=resolution[1] - 1))
        coord_idx8 = (coord_idx[0],
                      torch.clamp(coord_idx[1] - temp, min=0))

        image.index_put_(coord_idx8, mod_rgb, accumulate=False)
        image.index_put_(coord_idx7, mod_rgb, accumulate=False)
        image.index_put_(coord_idx6, mod_rgb, accumulate=False)
        image.index_put_(coord_idx5, mod_rgb, accumulate=False)
        image.index_put_(coord_idx4, mod_rgb, accumulate=False)
        image.index_put_(coord_idx3, mod_rgb, accumulate=False)
        image.index_put_(coord_idx2, mod_rgb, accumulate=False)
        image.index_put_(coord_idx1, mod_rgb, accumulate=False)
        image.index_put_(coord_idx, mod_rgb, accumulate=False)

        image = image * 255

        if not return_torch:
            image = image.cpu().numpy().astype(np.uint8)
    if return_coord:
        # mod_idx is in (i, j) format, not (x, y) format
        inv_mod_idx = torch.argsort(mod_idx)
        return image, save_coord_idx[inv_mod_idx]
    elif return_norm_coord:
        inv_mod_idx = torch.argsort(mod_idx)
        return image, orig_coord_idx[inv_mod_idx]
    else:
        return image


def make_depth(xyz: torch.Tensor, resolution: Tuple[int, int] = (200, 400), return_torch: bool = False) -> Union[torch.Tensor, np.array]:
    """
    Make depth image from xyz and rgb tensors

    Args:
        xyz: (N, 3) torch tensor containing xyz coordinates
        resolution: Tuple size of 2, returning panorama image of size resolution
        return_torch: if True, return image as torch.Tensor
                      if False, return image as numpy.array

    Returns:
        image: (H, W, 3) torch.Tensor or numpy.array
    """

    with torch.no_grad():

        # project farther points first
        dist = torch.norm(xyz, dim=-1)
        mod_idx = torch.argsort(dist)
        mod_idx = torch.flip(mod_idx, dims=[0])
        mod_xyz = xyz.clone().detach()[mod_idx]
        depth_val = torch.norm(mod_xyz, dim=-1)

        orig_coord_idx = cloud2idx(mod_xyz)
        coord_idx = (orig_coord_idx + 1.0) / 2.0
        # coord_idx[:, 0] is x coordinate, coord_idx[:, 1] is y coordinate
        coord_idx[:, 0] *= (resolution[1] - 1)
        coord_idx[:, 1] *= (resolution[0] - 1)

        coord_idx = torch.flip(coord_idx, [-1])
        coord_idx = coord_idx.long()
        coord_idx = tuple(coord_idx.t())

        image = torch.zeros([resolution[0], resolution[1]], dtype=torch.float, device=xyz.device)

        image.index_put_(coord_idx, depth_val, accumulate=False)

        if not return_torch:
            image = image.cpu().numpy()
    return image


def quantile(x: torch.Tensor, q: float) -> Tuple[Union[int, float], Union[int, float]]:
    """
    Obtain q quantile value and (1 - q) quantile value from x

    Args:
        x: 1-dim torch tensor
        q: q value for quantile

    Returns:
        result_1: q quantile value of x
        result_2: (1 - q) quantile value of x
    """

    with torch.no_grad():
        inds = torch.argsort(x)
        val_1 = int(len(x) * q)
        val_2 = int(len(x) * (1 - q))

        result_1 = x[inds[val_1]]
        result_2 = x[inds[val_2]]

    return result_1, result_2


def out_of_room(xyz: torch.Tensor, trans: torch.Tensor, out_quantile: float = 0.05) -> bool:
    """
    Check if translation is out of xyz coordinates

    Args:
        xyz: (N, 3) torch tensor containing xyz coordinates
        trans: (3, 1) torch tensor containing xyz translation

    Returns:
        False if translation is not out of room
        True if translation is out of room
    """

    with torch.no_grad():
        # rejecting outliers
        x_min, x_max = quantile(xyz[:, 0], out_quantile)
        y_min, y_max = quantile(xyz[:, 1], out_quantile)
        z_min, z_max = quantile(xyz[:, 2], out_quantile)

        if x_min < trans[0][0] < x_max and y_min < trans[1][0] < y_max and z_min < trans[2][0] < z_max:
            return False
        else:
            return True


def rot_from_ypr(ypr_array):
    def _ypr2mtx(ypr):
        # ypr is assumed to have a shape of [3, ]
        yaw, pitch, roll = ypr
        yaw = yaw.unsqueeze(0)
        pitch = pitch.unsqueeze(0)
        roll = roll.unsqueeze(0)

        tensor_0 = torch.zeros(1, device=yaw.device)
        tensor_1 = torch.ones(1, device=yaw.device)

        RX = torch.stack([
                        torch.stack([tensor_1, tensor_0, tensor_0]),
                        torch.stack([tensor_0, torch.cos(roll), -torch.sin(roll)]),
                        torch.stack([tensor_0, torch.sin(roll), torch.cos(roll)])]).reshape(3, 3)

        RY = torch.stack([
                        torch.stack([torch.cos(pitch), tensor_0, torch.sin(pitch)]),
                        torch.stack([tensor_0, tensor_1, tensor_0]),
                        torch.stack([-torch.sin(pitch), tensor_0, torch.cos(pitch)])]).reshape(3, 3)

        RZ = torch.stack([
                        torch.stack([torch.cos(yaw), -torch.sin(yaw), tensor_0]),
                        torch.stack([torch.sin(yaw), torch.cos(yaw), tensor_0]),
                        torch.stack([tensor_0, tensor_0, tensor_1])]).reshape(3, 3)

        R = torch.mm(RZ, RY)
        R = torch.mm(R, RX)

        return R
    
    if len(ypr_array.shape) == 1:
        return _ypr2mtx(ypr_array)
    else:
        tot_mtx = []
        for ypr in ypr_array:
            tot_mtx.append(_ypr2mtx(ypr))
        return torch.stack(tot_mtx)


def reshape_img_tensor(img: torch.Tensor, size: Tuple):
    # Note that size is (X, Y)
    cv_img = (img.cpu().numpy() * 255).astype(np.uint8)
    cv_img = cv2.resize(cv_img, size)
    cv_img = cv_img / 255.

    return torch.from_numpy(cv_img).float().to(img.device)


def img2colors(img: torch.Tensor):

    non_zero_coords = torch.nonzero(img.sum(-1) != 0)
    phi = non_zero_coords[:, 0] / img.shape[0] * np.pi

    img_colors = img[tuple(non_zero_coords.t())]

    return img_colors, torch.sin(phi)


def synthetic_mod_color(orig_img, cfg):
    # Synthetic illumination change
    if getattr(cfg, 'synth_const', None) is not None:
        orig_img = orig_img // cfg.synth_const
    if getattr(cfg, 'synth_gamma', None) is not None:
        orig_img = (((orig_img / 255.) ** cfg.synth_gamma) * 255).astype(np.uint8)
    if getattr(cfg, 'synth_wb', None):
        orig_img[..., 0] = (((orig_img[..., 0] / 255.) * cfg.synth_r) * 255).astype(np.uint8)
        orig_img[..., 1] = (((orig_img[..., 1] / 255.) * cfg.synth_g) * 255).astype(np.uint8)
        orig_img[..., 2] = (((orig_img[..., 2] / 255.) * cfg.synth_b) * 255).astype(np.uint8)

        orig_img[orig_img > 255] = 255

    return orig_img


def save_trajectory(xyz, rgb, trans_list, rot_list, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # trans_list is assumed to be shape (N, 3) and rot_list is assumed to be shape (N, 3, 3)
    for idx, (trans, rot) in enumerate(zip(trans_list, rot_list)):
        pano_img = cv2.cvtColor(make_pano((xyz - trans) @ rot.T, rgb), cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(save_dir, f'seq_{idx}.png'), pano_img)


def generate_trajectory(trans, rot, mode='circle', **kwargs):
    if mode == 'circle':
        radius = kwargs['radius']
        num_points = kwargs['num_points']
        trans_list = torch.cat([trans.T for _ in range(num_points)], dim=0)
        rot_list = torch.stack([rot for _ in range(num_points)], dim=0)

        theta = torch.linspace(0, 2 * np.pi, num_points, device=trans.device)
        trans_displacement = torch.stack([radius * torch.cos(theta), radius * torch.sin(theta), torch.zeros_like(theta)], dim=1)
        trans_list = trans_list + trans_displacement

    return trans_list, rot_list


def generate_trans_points(init_dict, xyz_lim, device):
    num_trans_per_meter = init_dict['num_trans_per_meter']
    num_x = ceil(2 * xyz_lim[0] * num_trans_per_meter)
    num_y = ceil(2 * xyz_lim[1] * num_trans_per_meter)
    num_z = ceil(2 * xyz_lim[2] * num_trans_per_meter)

    if num_x % 2 == 0:
        num_x += 1
    if num_y % 2 == 0:
        num_y += 1
    if num_z % 2 == 0:
        num_z += 1

    x_points = (2 * torch.arange(num_x, device=device).float() / (num_x - 1) - 1) * xyz_lim[0]
    y_points = (2 * torch.arange(num_y, device=device).float() / (num_y - 1) - 1) * xyz_lim[1]
    z_points = (2 * torch.arange(num_z, device=device).float() / (num_z - 1) - 1) * xyz_lim[2]

    if num_x == 1:
        x_points.fill_(0.)
    if num_y == 1:
        y_points.fill_(0.)
    if num_z == 1:
        z_points.fill_(0.)

    trans_coords = torch.meshgrid(x_points, y_points, z_points)
    trans = torch.stack([trans_coords[0].reshape(-1), trans_coords[1].reshape(-1), trans_coords[2].reshape(-1)], dim=0).t()

    return trans

# Code excerpted from https://github.com/haruishi43/equilib
def create_coordinate(h_out: int, w_out: int, device=torch.device('cpu')) -> np.ndarray:
    r"""Create mesh coordinate grid with height and width

    return:
        coordinate: numpy.ndarray
    """
    xs = torch.linspace(0, w_out - 1, w_out, device=device)
    theta = np.pi - xs * 2 * math.pi / w_out
    ys = torch.linspace(0, h_out - 1, h_out, device=device)
    phi = ys * math.pi / h_out
    # NOTE: https://github.com/pytorch/pytorch/issues/15301
    # Torch meshgrid behaves differently than numpy
    phi, theta = torch.meshgrid([phi, theta])
    coord = torch.stack((theta, phi), axis=-1)
    return coord


def compute_sampling_grid(ypr, num_split_h, num_split_w, inverse=False):
    """
    Utility function for computing sampling grid using yaw, pitch, roll
    We assume the equirectangular image to be splitted as follows:

    -------------------------------------
    |   0    |   1    |    2   |    3   |
    |        |        |        |        |
    -------------------------------------
    |   4    |   5    |    6   |    7   |
    |        |        |        |        |
    -------------------------------------

    Indices are assumed to be ordered in compliance to the above convention.
    Args:
        ypr: torch.tensor of shape (3, ) containing yaw, pitch, roll
        num_split_h: Number of horizontal splits
        num_split_w: Number of vertical splits
        inverse: If True, calculates sampling grid with inverted rotation provided from ypr

    Returns:
        grid: Sampling grid for generating rotated images according to yaw, pitch, roll
    """
    if inverse:
        R = rot_from_ypr(ypr)
    else:
        R = rot_from_ypr(ypr).T

    H, W = num_split_h, num_split_w
    a = create_coordinate(H, W, ypr.device)
    a[..., 0] -= np.pi / (num_split_w)  # Add offset to align sampling grid to each pixel center
    a[..., 1] += np.pi / (num_split_h * 2)  # Add offset to align sampling grid to each pixel center
    norm_A = 1
    x = norm_A * torch.sin(a[:, :, 1]) * torch.cos(a[:, :, 0])
    y = norm_A * torch.sin(a[:, :, 1]) * torch.sin(a[:, :, 0])
    z = norm_A * torch.cos(a[:, :, 1])
    A = torch.stack((x, y, z), dim=-1)  # (H, W, 3)
    _B = R @ A.unsqueeze(3)
    _B = _B.squeeze(3)
    grid = cloud2idx(_B.reshape(-1, 3)).reshape(H, W, 2)
    return grid


def generate_rot_points(init_dict=None, device='cpu'):
    """
    Generate rotation starting points

    Args:
        init_dict: Dictionary containing details of initialization
        device: Device in which rotation starting points will be saved

    Returns:
        rot_arr: (N, 3) array containing (yaw, pitch, roll) starting points
    """

    if init_dict['yaw_only']:
        rot_arr = torch.zeros(init_dict['num_yaw'], 3, device=device)
        rot = torch.arange(init_dict['num_yaw'], dtype=torch.float, device=device)
        rot = rot * 2 * np.pi / init_dict['num_yaw']
        rot_arr[:, 0] = rot

    else:
        # Perform 3 DoF initialization
        rot_coords = torch.meshgrid(torch.arange(init_dict['num_yaw'], device=device).float() / init_dict['num_yaw'],
            torch.arange(init_dict['num_pitch'], device=device).float() / init_dict['num_pitch'],
            torch.arange(init_dict['num_roll'], device=device).float() / init_dict['num_roll'])

        rot_arr = torch.stack([rot_coords[0].reshape(-1), rot_coords[1].reshape(-1), rot_coords[2].reshape(-1)], dim=0).t()

        rot_arr[:, 0] = (rot_arr[:, 0] * (init_dict['max_yaw'] - init_dict['min_yaw'])) + init_dict['min_yaw']
        rot_arr[:, 1] = (rot_arr[:, 1] * (init_dict['max_pitch'] - init_dict['min_pitch'])) + init_dict['min_pitch']
        rot_arr[:, 2] = (rot_arr[:, 2] * (init_dict['max_roll'] - init_dict['min_roll'])) + init_dict['min_roll']

        # Initialize grid sample locations
        grid_list = [compute_sampling_grid(ypr, init_dict['num_yaw'], init_dict['num_pitch']) for ypr in rot_arr]

        # Filter out overlapping rotations
        round_digit = 3
        rot_list = [str(np.around(grid.cpu().numpy(), round_digit)) for grid in grid_list]
        valid_rot_idx = [rot_list.index(rot_mtx) for rot_mtx in sorted(set(rot_list))]  # sorted added to make things deterministic
        rot_arr = torch.stack([rot_arr[idx] for idx in valid_rot_idx], dim=0)
        
        # Put identity at front
        zero_idx = torch.where(rot_arr.sum(-1) == 0.)[0].item()
        rot_arr[[0, zero_idx]] = rot_arr[[zero_idx, 0]]

    return rot_arr


def rand_trans_rot(max_trans=[0.2, 0.2, 0.2], max_theta=2 * np.pi, device='cpu'):
    # Return a random translation and rotation perturbation bounded by max_trans and max_theta
    trans_x = (2 * random.random() - 1) * max_trans[0]
    trans_y = (2 * random.random() - 1) * max_trans[1]
    trans_z = (2 * random.random() - 1) * max_trans[2]
    trans = torch.tensor([[trans_x, trans_y, trans_z]], device=device)  # (1, 3)
    if type(max_theta) is float:
        rot_theta = random.random() * max_theta
        rot = torch.tensor([[cos(rot_theta), -sin(rot_theta), 0], [sin(rot_theta), cos(rot_theta), 0], [0, 0, 1]], device=device)  # (3, 3)
    else:
        max_yaw, max_pitch, max_roll = max_theta
        rand_yaw = random.random() * max_yaw
        rand_pitch = random.random() * max_pitch
        rand_roll = random.random() * max_roll
        rot = rot_from_ypr(torch.tensor([rand_yaw, rand_pitch, rand_roll])).to(device)

    return trans, rot


def img_chunk(img: torch.Tensor, num_split_h, num_split_w):
    # Split img of shape H x W x C to a chunk tensor of shape num_split_h x num_split_w x Hc x Wc x C
    chunk_list = []
    for img_hor_chunk in torch.chunk(img, num_split_h, dim=0):
        chunk_list.append(torch.stack([*torch.chunk(img_hor_chunk, num_split_w, dim=1)], dim=0))
    chunk_tensor = torch.stack(chunk_list, dim=0)  # (num_split_h, num_split_w, Hc, Wc, C)
    return chunk_tensor


def img_dechunk(chunk_tensor: torch.Tensor):
    # Reconstruct an image of shape H x W x C from a chunk tensor of shape num_split_h x num_split_w x Hc x Wc x C
    assert len(chunk_tensor.shape) == 5
    num_split_h, num_split_w, Hc, Wc, C = chunk_tensor.shape
    new_chunk_tensor = chunk_tensor.permute(0, 2, 1, 3, 4)  # num_split_h x Hc x num_split_w x Wc x C
    new_chunk_tensor = new_chunk_tensor.reshape(num_split_h, Hc, -1, C)  # num_split_h x Hc x (num_split_w * Wc) x C
    img = new_chunk_tensor.reshape(-1, num_split_w * Wc, C)  # (num_split_h x Hc) x (num_split_w * Wc) x C

    return img


def marginal_pdf(
    values: torch.Tensor, bins: torch.Tensor, sigma: torch.Tensor, epsilon: float = 1e-10
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate the marginal probability distribution function of the input tensor based on the number of
    histogram bins. Excerpted from https://kornia.readthedocs.io/en/latest/_modules/kornia/enhance/histogram.html#histogram

    Args:
        values: shape [N].
        bins: shape [NUM_BINS].
        sigma: shape [1], gaussian smoothing factor.
        epsilon: scalar, for numerical stability.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
          - torch.Tensor: shape [BxN].
          - torch.Tensor: shape [BxNxNUM_BINS].
    """

    if not isinstance(values, torch.Tensor):
        raise TypeError(f"Input values type is not a torch.Tensor. Got {type(values)}")

    if not isinstance(bins, torch.Tensor):
        raise TypeError(f"Input bins type is not a torch.Tensor. Got {type(bins)}")

    if not isinstance(sigma, torch.Tensor):
        raise TypeError(f"Input sigma type is not a torch.Tensor. Got {type(sigma)}")

    if not values.dim() == 1:
        raise ValueError("Input values must be a of the shape N." " Got {}".format(values.shape))

    if not bins.dim() == 1:
        raise ValueError("Input bins must be a of the shape NUM_BINS" " Got {}".format(bins.shape))

    if not sigma.dim() == 0:
        raise ValueError("Input sigma must be a of the shape 1" " Got {}".format(sigma.shape))

    residuals = values.unsqueeze(-1) - bins.unsqueeze(0)
    kernel_values = torch.exp(-0.5 * (residuals / sigma).pow(2))
    pdf = torch.mean(kernel_values, dim=0)
    normalization = torch.sum(pdf) + epsilon
    pdf = pdf / normalization

    return pdf


def rgb_to_grayscale(
    image: torch.Tensor, rgb_weights: torch.Tensor = torch.tensor([0.299, 0.587, 0.114])
) -> torch.Tensor:
    r"""Convert a RGB image to grayscale version of image.

    .. image:: _static/img/rgb_to_grayscale.png

    The image data is assumed to be in the range of (0, 1).

    Args:
        image: RGB image to be converted to grayscale with shape :math:`(*,3,H,W)`.
        rgb_weights: Weights that will be applied on each channel (RGB).
            The sum of the weights should add up to one.
    Returns:
        grayscale version of the image with shape :math:`(*,1,H,W)`.

    .. note::
       See a working example `here <https://kornia-tutorials.readthedocs.io/en/latest/
       color_conversions.html>`__.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> gray = rgb_to_grayscale(input) # 2x1x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    if not isinstance(rgb_weights, torch.Tensor):
        raise TypeError(f"rgb_weights is not a torch.Tensor. Got {type(rgb_weights)}")

    if rgb_weights.shape[-1] != 3:
        raise ValueError(f"rgb_weights must have a shape of (*, 3). Got {rgb_weights.shape}")

    r: torch.Tensor = image[..., 0:1, :, :]
    g: torch.Tensor = image[..., 1:2, :, :]
    b: torch.Tensor = image[..., 2:3, :, :]

    if not torch.is_floating_point(image) and (image.dtype != rgb_weights.dtype):
        raise TypeError(
            f"Input image and rgb_weights should be of same dtype. Got {image.dtype} and {rgb_weights.dtype}"
        )

    w_r, w_g, w_b = rgb_weights.to(image).unbind()
    return w_r * r + w_g * g + w_b * b


def synthetic_mod_color(orig_img, cfg):
    # Synthetic illumination change
    if getattr(cfg, 'synth_const', None) is not None:
        orig_img = orig_img // cfg.synth_const
    if getattr(cfg, 'synth_gamma', None) is not None:
        orig_img = (((orig_img / 255.) ** cfg.synth_gamma) * 255).astype(np.uint8)
    if getattr(cfg, 'synth_wb', None):
        orig_img[..., 0] = (((orig_img[..., 0] / 255.) * cfg.synth_r) * 255).astype(np.uint8)
        orig_img[..., 1] = (((orig_img[..., 1] / 255.) * cfg.synth_g) * 255).astype(np.uint8)
        orig_img[..., 2] = (((orig_img[..., 2] / 255.) * cfg.synth_b) * 255).astype(np.uint8)

        orig_img[orig_img > 255] = 255
    if getattr(cfg, 'synth_noise', None):
        orig_img = 2 * (orig_img.astype(float) / 255.) - 1
        if getattr(cfg, 'noise_type', 'gaussian') in ['gaussian', 'speckle']:
            kwargs = {'var': getattr(cfg, 'var', 0.01)}
        elif getattr(cfg, 'noise_type', 'gaussian') in ['s&p', 'salt', 'pepper']:
            kwargs = {'amount': getattr(cfg, 'amount', 0.01)}
        else:
            kwargs = {}
        orig_img = random_noise(orig_img, getattr(cfg, 'noise_type', 'gaussian'), clip=True, **kwargs)
        orig_img = (((orig_img + 1) / 2.) * 255).astype(np.uint8)
    return orig_img
