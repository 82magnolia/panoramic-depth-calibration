import torch
import random
import da_utils
from depth_estimation import depth_utils
from utils import cloud2idx, ij2coord, rand_trans_rot
import numpy as np
from pytorch3d.ops import knn_points, estimate_pointcloud_normals


# Standard depth metrics imported from https://github.com/alibaba/UniFuse-Unidirectional-Fusion
def compute_depth_metrics(gt, pred, depth_thres=10, mask=None, median_align=False):
    """Computation of metrics between predicted and ground truth depths
    """

    if mask is None:
        mask = gt > 0
    else:
        mask = (gt > 0) & (mask)
    
    gt = gt[mask]
    pred = pred[mask]

    gt[gt<0.1] = 0.1
    pred[pred<0.1] = 0.1
    gt[gt>depth_thres] = depth_thres
    gt[pred>depth_thres] = depth_thres

    if median_align:
        pred *= torch.median(gt) / torch.median(pred)

    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log10(gt) - torch.log10(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    mae = torch.mean(torch.abs(gt - pred))

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)

    sq_rel = torch.mean((gt - pred) ** 2 / gt)

    error_dict = {'MAE': mae,
        'ABS_REL': abs_rel,
        'SQ_REL': sq_rel,
        'RMSE': rmse,
        'RMSE_LOG': rmse_log,
        'A1': a1,
        'A2': a2,
        'A3': a3}

    return error_dict


def stretch_loss(depth, depth_input, ref_depth_estimator, cfg):
    B = depth_input.shape[0]
    H, W = depth_input.shape[-2:]
    device = depth_input.device
    stretch_loss = 0.0
    k_list = [0.64, 0.8, 1.0, 1.25, 1.5625]
    # Depth weights for correcting depth from panorama stretches
    ij_values = torch.meshgrid(torch.arange(H), torch.arange(W))
    ij_sphere = ij2coord(torch.stack(ij_values, dim=-1).reshape(-1, 2), (H, W))

    # Stretch related attributes
    log_thres_high = getattr(cfg, 'log_thres_high', 0.9)
    log_thres_low = getattr(cfg, 'log_thres_low', 0.1)

    for idx in range(B):
        tgt_depth = None
        ref_depth_list = []
        valid_depth = (depth_input[idx].sum(0) != 0).unsqueeze(0)
        avg_log = torch.log(depth[idx][valid_depth]).mean()

        if avg_log > log_thres_high or avg_log < log_thres_low:
            for stretch_k in k_list:
                stretch_input = da_utils.pano_stretch(depth_input[idx].permute(1, 2, 0).cpu().detach().numpy(), stretch_k, stretch_k)
                stretch_input = torch.from_numpy(stretch_input).float().permute(2, 0, 1).unsqueeze(0).to(device)
                stretch_depth = depth_utils.inference(stretch_input, ref_depth_estimator, device, False)

                trans_ij_sphere = ij_sphere * torch.tensor([[stretch_k, stretch_k, 1]])
                trans_ij_sphere = trans_ij_sphere.norm(dim=-1).reshape(H, W, 1).to(device)

                ref_depth = da_utils.pano_stretch(stretch_depth.squeeze(0).permute(1, 2, 0).cpu().numpy(), 1 / stretch_k, 1 / stretch_k)
                ref_depth = torch.from_numpy(ref_depth).to(device)
                ref_depth = ref_depth * 1 / trans_ij_sphere
                ref_depth_list.append(ref_depth)
                
            tot_depth = torch.cat(ref_depth_list, dim=-1)  # (H, W, N_ref)

            if avg_log > log_thres_high:
                tgt_idx_list = list(range(len(k_list) // 2))
                tgt_depth_list = [tot_depth[..., min_idx: min_idx+1] for min_idx in tgt_idx_list]
            elif avg_log < log_thres_low:
                tgt_idx_list = list(range(len(k_list) // 2 + 1, len(k_list)))
                tgt_depth_list = [tot_depth[..., min_idx: min_idx+1] for min_idx in tgt_idx_list]
            else:
                min_idx = len(k_list) // 2
                tgt_depth_list = [tot_depth[..., min_idx: min_idx+1]]

            num_tgt = len(tgt_depth_list)
            
            stretch_loss_update = 0.0
            for tgt_depth in tgt_depth_list:
                stretch_loss_update += ((depth[idx][valid_depth] - tgt_depth.permute(2, 0, 1)[valid_depth]) ** 2).mean().sqrt()
            stretch_loss_update /= num_tgt
            
            stretch_loss += stretch_loss_update / B

    return stretch_loss


def synth_view_loss(depth, depth_input, depth_estimator, cfg):
    # Assume all image tensors are of shape (B, C, H, W)
    B, _, H, W = depth.shape
    max_trans = getattr(cfg, 'synth_max_trans', [0.5, 0.5, 0.5])
    max_theta = getattr(cfg, 'synth_max_theta', 2 * np.pi)
    loss_val = 0.0

    for idx in range(B):
        depth_pcd = depth_utils.get_point_cloud(depth[idx], return_torch=True)  # (N_d, 3)
        rand_trans, rand_rot = rand_trans_rot(max_trans, max_theta, depth.device)
        transform_pcd = (depth_pcd - rand_trans) @ rand_rot.T
        if cfg.depth_method != 'UNet':
            valid_depth = depth_pcd.norm(dim=-1) > 0.

        # Image coordinates for projected point cloud
        norm_coords = cloud2idx(transform_pcd)  # (N_d, 2)
        coords = (norm_coords + 1) / 2.
        coords[:, 0] *= W
        coords[:, 1] *= H
        coords -= 0.5
        coords = coords.reshape(H, W, 2).permute(2, 0, 1).unsqueeze(0)

        # Generate depth maps for synthetic views
        with torch.no_grad():
            synth_view, mask_view = depth_utils.render(depth_input[idx: idx+1], depth[idx: idx+1], coords)
        synth_depth = depth_utils.inference(synth_view, depth_estimator, depth.device, True)
        synth_depth_pcd = depth_utils.get_point_cloud(synth_depth, return_torch=True)  # (N_d, 3)
        mask_total = mask_view.reshape(-1)

        nn_size = 1
        if cfg.depth_method != 'UNet':
            nn12 = knn_points(transform_pcd[valid_depth].unsqueeze(0), synth_depth_pcd[mask_total].unsqueeze(0), K=nn_size)
        else:
            nn12 = knn_points(transform_pcd.unsqueeze(0), synth_depth_pcd[mask_total].unsqueeze(0), K=nn_size)
        nn_dists = nn12.dists.squeeze(0)[:, 0]
        cd_val = nn_dists[nn_dists < 0.5].mean()
        loss_val += 2 * cd_val / B
    return loss_val


def normal_loss(depth, depth_input, depth_estimator, cfg):
    # Assume all image tensors are of shape (B, C, H, W)
    B, _, H, W = depth.shape
    max_trans = getattr(cfg, 'synth_max_trans', [0.5, 0.5, 0.5])
    max_theta = getattr(cfg, 'synth_max_theta', 2 * np.pi)
    loss_val = 0.0

    for idx in range(B):
        depth_pcd = depth_utils.get_point_cloud(depth[idx], return_torch=True)  # (N_d, 3)
        if cfg.depth_method != 'UNet':
            valid_depth = depth_pcd.norm(dim=-1) > 0.
        rand_trans, rand_rot = rand_trans_rot(max_trans, max_theta, depth.device)
        transform_pcd = (depth_pcd - rand_trans) @ rand_rot.T
        
        # Image coordinates for projected point cloud
        norm_coords = cloud2idx(transform_pcd)  # (N_d, 2)
        coords = (norm_coords + 1) / 2.
        coords[:, 0] *= W
        coords[:, 1] *= H
        coords -= 0.5
        coords = coords.reshape(H, W, 2).permute(2, 0, 1).unsqueeze(0)

        # Generate depth maps for synthetic views
        with torch.no_grad():
            synth_view, mask_view = depth_utils.render(depth_input[idx: idx+1], depth[idx: idx+1], coords)
        synth_depth = depth_utils.inference(synth_view, depth_estimator, depth.device, True)
        synth_depth_pcd = depth_utils.get_point_cloud(synth_depth, return_torch=True)  # (N_d, 3)
        mask_total = mask_view.reshape(-1)

        nn_size = 15
        if cfg.depth_method != 'UNet':
            nn12 = knn_points(transform_pcd[valid_depth].unsqueeze(0), synth_depth_pcd[mask_total].unsqueeze(0), K=nn_size)
        else:
            nn12 = knn_points(transform_pcd.unsqueeze(0), synth_depth_pcd[mask_total].unsqueeze(0), K=nn_size)
        nn_dists = nn12.dists.squeeze(0)[:, 0]

        # Impose normal loss
        if cfg.depth_method != 'UNet':
            transform_normals = estimate_pointcloud_normals(transform_pcd[valid_depth].unsqueeze(0), neighborhood_size=nn_size).squeeze()            
            nn_pcd = synth_depth_pcd[mask_total][nn12.idx.squeeze()]
            norm_dist = (transform_normals.unsqueeze(1) * (nn_pcd - transform_pcd[valid_depth].unsqueeze(1))).sum(dim=-1).abs()
        else:
            transform_normals = estimate_pointcloud_normals(transform_pcd.unsqueeze(0), neighborhood_size=nn_size).squeeze()            
            nn_pcd = synth_depth_pcd[mask_total][nn12.idx.squeeze()]
            norm_dist = (transform_normals.unsqueeze(1) * (nn_pcd - transform_pcd.unsqueeze(1))).sum(dim=-1).abs()
        norm_val = norm_dist[nn_dists < 0.5].mean()
        loss_val += 2 * norm_val / B
    return loss_val


def generate_synth_view(depth, depth_input, depth_estimator, augment_dict, cfg, sample_augment=None, \
        return_scale_depth=False, depth_scale=None):
    # Assume all image tensors are of shape (B, C, H, W)
    B, _, H, W = depth.shape
    device = depth.device
    if sample_augment is None:
        sample_augment = [True for _ in range(B)]

    max_trans = augment_dict['max_trans']
    max_theta = augment_dict['max_theta']

    log_thres_high = getattr(cfg, 'log_thres_high', 0.9)
    log_thres_low = getattr(cfg, 'log_thres_low', 0.1)

    synth_view_list = []
    synth_depth_list = []
    synth_scaled_list = []

    # Depth weights for correcting depth from panorama stretches
    ij_values = torch.meshgrid(torch.arange(H), torch.arange(W))
    ij_sphere = ij2coord(torch.stack(ij_values, dim=-1).reshape(-1, 2), (H, W))

    for idx in range(B):
        if sample_augment[idx]:
            depth_pcd = depth_utils.get_point_cloud(depth[idx], return_torch=True)  # (N_d, 3)
            valid_depth = (depth_input[idx].sum(0) != 0).unsqueeze(0)
            avg_log = torch.log(depth[idx][valid_depth]).mean()

            # Conditionally select random motion
            if avg_log < log_thres_high and avg_log > log_thres_low:
                rand_trans, rand_rot = rand_trans_rot(max_trans, max_theta, depth.device)
                transform_pcd = (depth_pcd - rand_trans) @ rand_rot.T

                # Image coordinates for projected point cloud
                norm_coords = cloud2idx(transform_pcd)  # (N_d, 2)
                coords = (norm_coords + 1) / 2.
                coords[:, 0] *= W
                coords[:, 1] *= H
                coords -= 0.5
                coords = coords.reshape(H, W, 2).permute(2, 0, 1).unsqueeze(0)

                # Generate depth maps for synthetic views
                with torch.no_grad():
                    synth_view, mask_view = depth_utils.render(depth_input[idx: idx+1], depth[idx: idx+1], coords)
                    if return_scale_depth:
                        synth_scaled_depth, mask_depth = depth_utils.render(depth[idx: idx+1], depth[idx: idx+1], coords)
                        synth_scaled_list.append(synth_scaled_depth / synth_scaled_depth.max())
            else:
                # Conditionally choose stretch values
                if avg_log > log_thres_high:  # Augment with larger scenes to aid stretch loss training
                    k_min = 1.953125
                    k_max = 2.44140625
                    stretch_k = random.random() * (k_max - k_min) + k_min
                else:
                    k_min = 0.4096
                    k_max = 0.512
                    stretch_k = random.random() * (k_max - k_min) + k_min

                synth_view = da_utils.pano_stretch(depth_input[idx].permute(1, 2, 0).cpu().detach().numpy(), stretch_k, stretch_k)
                synth_view = torch.from_numpy(synth_view).float().permute(2, 0, 1).unsqueeze(0).to(device)
                if return_scale_depth:
                    stretch_depth = da_utils.pano_stretch(depth[idx].permute(1, 2, 0).cpu().detach().numpy(), stretch_k, stretch_k)
                    stretch_depth = torch.from_numpy(stretch_depth).to(device)

                    trans_ij_sphere = ij_sphere * torch.tensor([[stretch_k, stretch_k, 1]])
                    trans_ij_sphere = trans_ij_sphere.norm(dim=-1).reshape(H, W, 1).to(device)
                    synth_scaled_depth = stretch_depth * trans_ij_sphere
                    synth_scaled_depth = synth_scaled_depth.permute(2, 0, 1).unsqueeze(0)
                    synth_scaled_list.append(synth_scaled_depth / synth_scaled_depth.max())

            synth_depth = depth_utils.inference(synth_view, depth_estimator, depth.device, True)
            synth_view_list.append(synth_view)
            synth_depth_list.append(synth_depth)
        else:
            synth_view_list.append(depth_input[idx: idx+1])
            synth_depth_list.append(depth[idx: idx+1])
            if return_scale_depth:
                synth_scaled_list.append(depth[idx: idx+1] / depth_scale[idx])
    
    synth_view = torch.cat(synth_view_list, dim=0)
    synth_depth = torch.cat(synth_depth_list, dim=0)

    if return_scale_depth:
        synth_scaled_depth = torch.cat(synth_scaled_list, dim=0)
        return synth_depth, synth_view, synth_scaled_depth
    else:
        return synth_depth, synth_view
