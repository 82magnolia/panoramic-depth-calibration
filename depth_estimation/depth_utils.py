import torch
from depth_estimation.omnidepth.utils.spherical import Spherical
from depth_estimation.omnidepth.utils.spherical_deprojection import SphericalDeprojection
import importlib
import numpy as np
from scipy.ndimage import distance_transform_edt


def inference(input, model, device, with_grad, return_ft=False):
    if with_grad:
        if return_ft:
            depth, ft = model(input, True)
        else:
            depth = model(input)
    else:
        with torch.no_grad():
            if return_ft:
                depth, ft = model(input, True)
            else:
                depth = model(input)
    if return_ft:
        return depth, ft
    else:
        return depth

def get_point_cloud(depth, return_torch=False):
    device = depth.get_device()
    if device == -1:
        device = 'cpu'
    sgrid = Spherical(width=512,mode='pi',long_offset_pi=-0.5).to(device)(depth)
    pcloud = SphericalDeprojection().to(device)(depth,sgrid)
    pred_xyz = pcloud[0]

    if return_torch:
        pred_xyz = pred_xyz.reshape(3, -1).T
    else:
        pred_xyz = pred_xyz.reshape(3, -1).cpu().numpy().T

    # Rotation matrices for aligning rotations with the panorama
    rot_x = np.array([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]])
    rot_z = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])

    if return_torch:
        rot_x = torch.from_numpy(rot_x).float().to(device)
        rot_z = torch.from_numpy(rot_z).float().to(device)

    pred_xyz = pred_xyz @ rot_x.T @ rot_z.T
    return pred_xyz

def get_estimator(method, device, **kwargs):
    method_module = importlib.import_module(f'depth_estimation.{method.lower()}')

    return method_module.get_model(device, **kwargs).eval()

def get_preprocessor(method):
    method_module = importlib.import_module(f'depth_estimation.{method.lower()}')

    return method_module.preprocess


# Point splatting code excerpted from https://github.com/VCL3D/SphericalViewSynthesis

def __splat__(values, coords, splatted):
    b, c, h, w = splatted.size()
    uvs = coords
    u = uvs[:, 0, :, :].unsqueeze(1)
    v = uvs[:, 1, :, :].unsqueeze(1)
    
    u0 = torch.floor(u)
    u1 = u0 + 1
    v0 = torch.floor(v)
    v1 = v0 + 1

    u0_safe = torch.clamp(u0, 0.0, w-1)
    v0_safe = torch.clamp(v0, 0.0, h-1)
    u1_safe = torch.clamp(u1, 0.0, w-1)
    v1_safe = torch.clamp(v1, 0.0, h-1)

    u0_w = (u1 - u) * (u0 == u0_safe).detach().type(values.dtype)
    u1_w = (u - u0) * (u1 == u1_safe).detach().type(values.dtype)
    v0_w = (v1 - v) * (v0 == v0_safe).detach().type(values.dtype)
    v1_w = (v - v0) * (v1 == v1_safe).detach().type(values.dtype)

    top_left_w = u0_w * v0_w
    top_right_w = u1_w * v0_w
    bottom_left_w = u0_w * v1_w
    bottom_right_w = u1_w * v1_w

    weight_threshold = 1e-3
    top_left_w *= (top_left_w >= weight_threshold).detach().type(values.dtype)
    top_right_w *= (top_right_w >= weight_threshold).detach().type(values.dtype)
    bottom_left_w *= (bottom_left_w >= weight_threshold).detach().type(values.dtype)
    bottom_right_w *= (bottom_right_w >= weight_threshold).detach().type(values.dtype)

    for channel in range(c):
        top_left_values = values[:, channel, :, :].unsqueeze(1) * top_left_w
        top_right_values = values[:, channel, :, :].unsqueeze(1) * top_right_w
        bottom_left_values = values[:, channel, :, :].unsqueeze(1) * bottom_left_w
        bottom_right_values = values[:, channel, :, :].unsqueeze(1) * bottom_right_w

        top_left_values = top_left_values.reshape(b, -1)
        top_right_values = top_right_values.reshape(b, -1)
        bottom_left_values = bottom_left_values.reshape(b, -1)
        bottom_right_values = bottom_right_values.reshape(b, -1)

        top_left_indices = (u0_safe + v0_safe * w).reshape(b, -1).type(torch.int64)
        top_right_indices = (u1_safe + v0_safe * w).reshape(b, -1).type(torch.int64)
        bottom_left_indices = (u0_safe + v1_safe * w).reshape(b, -1).type(torch.int64)
        bottom_right_indices = (u1_safe + v1_safe * w).reshape(b, -1).type(torch.int64)
        
        splatted_channel = splatted[:, channel, :, :].unsqueeze(1)
        splatted_channel = splatted_channel.reshape(b, -1)
        splatted_channel.scatter_add_(1, top_left_indices, top_left_values)
        splatted_channel.scatter_add_(1, top_right_indices, top_right_values)
        splatted_channel.scatter_add_(1, bottom_left_indices, bottom_left_values)
        splatted_channel.scatter_add_(1, bottom_right_indices, bottom_right_values)
    splatted = splatted.reshape(b, c, h, w)


def __weighted_average_splat__(depth, weights, epsilon=1e-8):
    zero_weights = (weights <= epsilon).detach().type(depth.dtype)
    return depth / (weights + epsilon * zero_weights)


def __depth_distance_weights__(depth, max_depth=20.0):
    weights = 1.0 / torch.exp(2 * depth / max_depth)
    return weights


def render(img, depth, coords, max_depth=20.0):
    splatted_img = torch.zeros_like(img)
    splatted_wgts = torch.zeros_like(depth)        
    weights = __depth_distance_weights__(depth, max_depth=max_depth)
    __splat__(img * weights, coords, splatted_img)
    __splat__(weights, coords, splatted_wgts)
    recon = __weighted_average_splat__(splatted_img, splatted_wgts)
    mask = (splatted_wgts > 1e-3).detach()
    return recon, mask


def render_with_fill(img, depth, coords, max_depth=20.0):
    splatted_img = torch.zeros_like(img)
    splatted_wgts = torch.zeros_like(depth)        
    weights = __depth_distance_weights__(depth, max_depth=max_depth)
    __splat__(img * weights, coords, splatted_img)
    __splat__(weights, coords, splatted_wgts)
    recon = __weighted_average_splat__(splatted_img, splatted_wgts)
    recon = recon.squeeze(0).cpu().permute(1, 2, 0).numpy()
    fill_idx = distance_transform_edt((recon.sum(-1) == 0), return_distances=False, return_indices=True)
    recon = torch.from_numpy(recon[tuple(fill_idx)]).to(depth.device)
    recon = recon.permute(2, 0, 1).unsqueeze(0)

    return recon


def render_to(src, tgt, wgts, depth, coords, max_depth=20.0):    
    weights = __depth_distance_weights__(depth, max_depth=max_depth)
    __splat__(src * weights, coords, tgt)
    __splat__(weights, coords, wgts)
    tgt = __weighted_average_splat__(tgt, wgts)
    mask = (wgts > 1e-3).detach()
    return mask
