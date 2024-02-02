from dataset import DataContainer
from depth_estimation import depth_utils
import torch
import torch.optim as optim
import os
from log_utils import DepthLogger, save_logger
import numpy as np
import cv2
from train_utils import (
    compute_depth_metrics,
    stretch_loss,
    synth_view_loss,
    generate_synth_view,
    normal_loss,
)


class Trainer():
    def __init__(self, cfg, log_dir):
        self.cfg = cfg
        self.batch_size = cfg.batch_size
        self.log_dir = log_dir
        self.logger = DepthLogger(self.log_dir)
        self.mode = cfg.mode
        self.data_container = DataContainer(cfg)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        depth_method = getattr(self.cfg, 'depth_method', 'UNet')

        if self.mode == 'train':
            if self.cfg.load_model is not None:
                print(f"Loading model from {self.cfg.load_model}")
                self.depth_estimator = torch.load(self.cfg.load_model).eval().to(self.device)
                self.ref_depth_estimator = torch.load(self.cfg.load_model).eval().to(self.device)
            else:
                self.depth_estimator = depth_utils.get_estimator(depth_method, self.device, max_depth=getattr(cfg, 'max_depth', 10.))
                self.ref_depth_estimator = depth_utils.get_estimator(depth_method, self.device, max_depth=getattr(cfg, 'max_depth', 10.))
            for param in self.ref_depth_estimator.parameters():
                param.requires_grad = False

        else:
            if self.cfg.load_model is not None:
                print(f"Loading model from {self.cfg.load_model}")
                self.depth_estimator = torch.load(self.cfg.load_model).eval().to(self.device)
            else:
                self.depth_estimator = depth_utils.get_estimator(depth_method, self.device, max_depth=getattr(cfg, 'max_depth', 10.))

        self.epochs = cfg.epochs
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.depth_estimator.parameters()), lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay)

        # Stretch related attributes
        self.std_threshold = getattr(cfg, 'std_threshold', 1.25)  # Threshold for selecting when to optimize stretches
        self.augment_mode = getattr(cfg, 'augment_mode', None)
        self.augment_dict = {'max_trans': getattr(cfg, 'synth_max_trans', [0.5, 0.5, 0.5]),
            'max_theta': getattr(cfg, 'synth_max_theta', 2 * np.pi)}
        self.update_ref_every = getattr(cfg, 'update_ref_every', 0)

    def run(self):
        if self.mode == 'train':
            print("Begin Train!")
            self.run_train()
        else:
            print("Begin Test!")
            self.run_test()
        
        save_logger(os.path.join(self.log_dir, self.cfg.log_name), self.logger)

    def run_train(self):
        for idx in range(self.epochs):
            self.set_augment(idx)
            self.depth_estimator.to(self.device)
            print(f"Epoch {idx} Training")
            self.train(idx)
            print(f"Epoch {idx} Evaluation")
            self.eval(idx)
            self.update_ref_estimator()  # Update reference estimator after each epoch
            torch.save(self.depth_estimator.to('cpu'), os.path.join(self.log_dir, f'model.pth'))  # Save latest model
        self.logger.convert_np()

    def run_test(self):
        self.eval(0)
        self.logger.convert_np()

    def train(self, epoch):
        epoch_metrics = {'MAE': 0., 'ABS_REL': 0., 'SQ_REL': 0., 'RMSE': 0., 'RMSE_LOG': 0., 'A1': 0., 'A2': 0., 'A3': 0.}
        data_count = 0
        for batch_idx, data_dict in enumerate(self.data_container.train_loader):
            self.optimizer.zero_grad()
            depth_input = data_dict['img'].to(self.device)
            gt_depth = data_dict['depth'].to(self.device)
            B = depth_input.shape[0]
            H, W = depth_input.shape[-2:]
            depth = depth_utils.inference(depth_input, self.depth_estimator, self.device, True)  # (B, H_d, W_d)
            loss = 0.0
            data_count += B

            # Calculate metrics
            with torch.no_grad():
                new_error_metrics = compute_depth_metrics(gt_depth, depth, getattr(self.cfg, 'depth_thres', 10))                

            if getattr(self.cfg, 'augment_data', False):
                sample_augment = data_dict['augment']
                depth, depth_input = generate_synth_view(depth, depth_input, self.depth_estimator, self.augment_dict, self.cfg, sample_augment)

            if getattr(self.cfg, 'stretch_loss', 0) > 0:                
                loss += self.cfg.stretch_loss * stretch_loss(depth, depth_input, self.ref_depth_estimator, self.cfg)
            if getattr(self.cfg, 'synth_view_loss', 0) > 0:
                loss += self.cfg.synth_view_loss * synth_view_loss(depth, depth_input, self.depth_estimator, self.cfg)
            if getattr(self.cfg, 'normal_loss', 0) > 0:
                loss += self.cfg.normal_loss * normal_loss(depth, depth_input, self.depth_estimator, self.cfg)

            for k, v in new_error_metrics.items():
                epoch_metrics[k] = (epoch_metrics[k] * batch_idx * self.batch_size + new_error_metrics[k].item() * B) / (batch_idx * self.batch_size + B)

            print_dict = {
                'Iter': batch_idx,
                'Loss': loss.item() if isinstance(loss, torch.Tensor) else 0.0,
                **epoch_metrics
            }

            if isinstance(loss, torch.Tensor) and loss.requires_grad:  # If loss is added and requires gradients
                loss.backward()
                self.optimizer.step()

            self.print_state(print_dict)
            self.logger.add_metric('train', epoch, **epoch_metrics)

        self.logger.set_data_count('train', data_count)

    def eval(self, epoch):
        epoch_metrics = {'MAE': 0., 'ABS_REL': 0., 'SQ_REL': 0., 'RMSE': 0., 'RMSE_LOG': 0., 'A1': 0., 'A2': 0., 'A3': 0.}
        data_count = 0
        for batch_idx, data_dict in enumerate(self.data_container.test_loader):
            depth_input = data_dict['img'].to(self.device)
            gt_depth = data_dict['depth'].to(self.device)
            B = depth_input.shape[0]
            H, W = depth_input.shape[-2:]
            data_count += B
            depth = depth_utils.inference(depth_input, self.depth_estimator, self.device, False)  # (B, H_d, W_d)
            # Mask top and bottom for networks other than UNet
            if self.cfg.depth_method != 'UNet':
                depth[..., :H//8, :] = 0
                depth[..., -H//8:, :] = 0 
                depth_input[..., :H//8, :] = 0 
                depth_input[..., -H//8:, :] = 0 
                mask = torch.zeros_like(depth).bool()
                mask[..., H//8: - H//8, :] = True

            # Calculate metrics
            with torch.no_grad():
                new_error_metrics = compute_depth_metrics(gt_depth, depth, getattr(self.cfg, 'depth_thres', 10))                

            for k, v in new_error_metrics.items():
                epoch_metrics[k] = (epoch_metrics[k] * batch_idx * self.batch_size + new_error_metrics[k].item() * B) / (batch_idx * self.batch_size + B)

            print_dict = {
                'Iter': batch_idx,
                **epoch_metrics
            }
            self.print_state(print_dict)
            self.logger.add_metric('eval', epoch, **epoch_metrics)

        self.logger.set_data_count('eval', data_count)

    def print_state(self, print_dict: dict):
        """
        Print current training state using values from print_dict.

        Args:
            print_dict: Dictionary containing arguments to print
        """
        print_str = ""
        for idx, key in enumerate(print_dict.keys()):
            if idx == len(print_dict.keys()) - 1:
                if type(print_dict[key]) == float:
                    print_str += f"{key} = {print_dict[key]:.4f}"
                else:
                    print_str += f"{key} = {print_dict[key]}"
            else:
                if type(print_dict[key]) == float:
                    print_str += f"{key} = {print_dict[key]:.4f}, "
                else:
                    print_str += f"{key} = {print_dict[key]}, "

        print(print_str)

    def set_augment(self, epoch):
        if self.augment_mode == 'constant':
            if epoch > 0:
                print("Setting augment dictionary to...")
                self.print_state(self.augment_dict)
        elif self.augment_mode == 'linear_scale':
            if epoch > 0:
                trans_increment = getattr(self.cfg, 'trans_increment', 0.1)
                self.augment_dict['max_trans'] = [val + trans_increment for val in self.augment_dict['max_trans']]
                print("Setting augment dictionary to...")
                self.print_state(self.augment_dict)

    def update_ref_estimator(self):
        print("Updating Reference Estimator...")
        self.ref_depth_estimator.load_state_dict(self.depth_estimator.state_dict())
