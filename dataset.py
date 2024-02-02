import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import data_utils
import utils
import cv2
from tqdm import tqdm
from utils import synthetic_mod_color


def read_stanford(img_file, cfg):
    depth_list = []
    valid_list = []
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    pbar = tqdm(total=len(img_file))
    past_pcd_name = ""
    for idx, filename in enumerate(img_file):
        split_type = filename.split('/')[-2]
        img_name = filename.split('/')[-1]
        room_type = img_name.split('_')[2]
        room_no = img_name.split('_')[3]
        area_num = int(filename.split('/')[-2].split('_')[-1])

        pcd_name = data_utils.get_pcd_name('stanford', area_name=split_type, room_type=room_type, room_no=room_no)
        if past_pcd_name != pcd_name:
            xyz_np, _ = data_utils.read_pcd('stanford', pcd_name=pcd_name, sample_rate=1)
            xyz = torch.from_numpy(xyz_np).float().to(device)
        past_pcd_name = pcd_name

        gt_trans, gt_rot = data_utils.read_gt('stanford', area_num=area_num, img_name=img_name)
        gt_trans = torch.from_numpy(gt_trans).float().to(device)
        gt_rot = torch.from_numpy(gt_rot).float().to(device)
        
        if utils.out_of_room(xyz, gt_trans):
            pbar.update()
            continue
        else:
            valid_list.append(idx)
            gt_xyz = (xyz - gt_trans.T) @ gt_rot.T
            depth_list.append(utils.make_depth(gt_xyz, (cfg.height, cfg.width), True).cpu())
            pbar.update()
    pbar.close()
    depth_arr = torch.stack(depth_list, dim=0)
    return depth_arr, valid_list


def read_omniscenes(img_file, cfg):
    depth_list = []
    valid_list = []
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    pbar = tqdm(total=len(img_file))
    past_pcd_name = ""
    for idx, filename in enumerate(img_file):
        video_name = filename.split('/')[-2]
        room_type = video_name.split('_')[1]
        room_no = video_name.split('_')[2]

        pcd_name = data_utils.get_pcd_name('omniscenes', room_type=room_type, room_no=room_no)
        if past_pcd_name != pcd_name:
            xyz_np, _ = data_utils.read_pcd('omniscenes', pcd_name=pcd_name, sample_rate=1)
            xyz = torch.from_numpy(xyz_np).float().to(device)
        past_pcd_name = pcd_name

        gt_trans, gt_rot = data_utils.read_gt('omniscenes', filename=filename)
        gt_trans = torch.from_numpy(gt_trans).float().to(device)
        gt_rot = torch.from_numpy(gt_rot).float().to(device)
        
        if utils.out_of_room(xyz, gt_trans):
            pbar.update()
            continue
        else:
            valid_list.append(idx)
            gt_xyz = (xyz - gt_trans.T) @ gt_rot.T
            depth_list.append(utils.make_depth(gt_xyz, (cfg.height, cfg.width), True).cpu())
            pbar.update()
    pbar.close()
    depth_arr = torch.stack(depth_list, dim=0)
    return depth_arr, valid_list


def depth_collate_fn(list_data):
    imgs, depths, idxs = list(zip(*list_data))
    img_batch = torch.stack(imgs, dim=0)
    if None in depths:
        return {
            'img': img_batch,
            'depth': None,
            'idx': idxs,
        }

    else:
        depth_batch = torch.stack(depths, dim=0)
        return {
            'img': img_batch,
            'depth': depth_batch,
            'idx': idxs,
        }


def depth_augment_collate_fn(list_data):
    imgs, depths, idxs, augments = list(zip(*list_data))
    img_batch = torch.stack(imgs, dim=0)
    if None in depths:
        return {
            'img': img_batch,
            'depth': None,
            'idx': idxs,
            'augment' : augments
        }
    else:
        depth_batch = torch.stack(depths, dim=0)
        return {
            'img': img_batch,
            'depth': depth_batch,
            'idx': idxs,
            'augment': augments
        }


class PanoDepthDataset(Dataset):
    def __init__(self, cfg, img_file_list=None):
        super(PanoDepthDataset, self).__init__()
        self.cfg = cfg
        self.height = cfg.height
        self.width = cfg.width
        self.dataset = cfg.dataset
        self.split_type = getattr(cfg, 'split_type', None)
        self.room_type = getattr(cfg, 'room_type', None)
        if img_file_list is None:
            self.img_file_list = data_utils.get_filename(self.dataset, split_type=self.split_type, room_type=self.room_type)
        else:
            self.img_file_list = img_file_list

        if self.dataset == 'stanford':
            self.depth_reader = read_stanford
        elif self.dataset == 'omniscenes':
            self.depth_reader = read_omniscenes

        self.target_domain = getattr(self.cfg, 'target_domain', 'default')
        self.img_file_list = [f for f in self.img_file_list]
        self.depth_arr, self.valid_list = self.depth_reader(self.img_file_list, cfg)

    def __len__(self):
        return len(self.depth_arr)
    
    def __getitem__(self, idx):
        tgt_idx = self.valid_list[idx]
        img = cv2.cvtColor(cv2.imread(self.img_file_list[tgt_idx]), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, dsize=(self.width, self.height))
        if self.target_domain == 'mod_color':
            img = synthetic_mod_color(img, self.cfg)
        img = torch.from_numpy(img).float() / 255.
        img = img.permute(2, 0, 1)
        depth = self.depth_arr[idx].unsqueeze(0)
        return img, depth, idx


class PanoAugmentDataset(Dataset):
    def __init__(self, cfg, pano_dataset: PanoDepthDataset):
        self.augment_factor = getattr(cfg, 'augment_factor', 1)
        print(f"Augmenting data by factor {self.augment_factor}...")
        self.orig_len = len(pano_dataset)
        self.augment_list = [False for _ in pano_dataset] + [True for _ in range(self.augment_factor * self.orig_len)]
        self.orig_dataset = pano_dataset

    def __getitem__(self, idx):
        orig_idx = idx % self.orig_len
        img, depth, _ = self.orig_dataset[orig_idx]
        augment = self.augment_list[idx]

        return img, depth, idx, augment

    def __len__(self):
        return len(self.augment_list)


class DataContainer():
    def __init__(self, cfg):
        if getattr(cfg, 'augment_data', False):
            orig_dataset = PanoDepthDataset(cfg)
            self.dataset = PanoAugmentDataset(cfg, orig_dataset)
            self.cfg = cfg
            self.train_loader = DataLoader(self.dataset, batch_size=cfg.batch_size, collate_fn=depth_augment_collate_fn,
                shuffle=True, num_workers=cfg.num_workers, drop_last=False, pin_memory=cfg.pin_memory)
            self.test_loader = DataLoader(orig_dataset, batch_size=cfg.batch_size, collate_fn=depth_collate_fn,
                shuffle=False, num_workers=cfg.num_workers, drop_last=False, pin_memory=cfg.pin_memory)
        else:
            self.dataset = PanoDepthDataset(cfg)
            self.cfg = cfg
            self.train_loader = DataLoader(self.dataset, batch_size=cfg.batch_size, collate_fn=depth_collate_fn,
                shuffle=True, num_workers=cfg.num_workers, drop_last=False, pin_memory=cfg.pin_memory)
            self.test_loader = DataLoader(self.dataset, batch_size=cfg.batch_size, collate_fn=depth_collate_fn,
                shuffle=False, num_workers=cfg.num_workers, drop_last=False, pin_memory=cfg.pin_memory)
