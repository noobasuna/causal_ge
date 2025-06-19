import os
import sys
import os.path as osp
import argparse
import cv2
import numpy as np
from glob import glob
import random
import time
import csv
import h5py
import copy
import pandas as pd
import importlib
from functools import partial
from omegaconf import OmegaConf
from tqdm import tqdm
from rich.progress import track
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

import torchvision
from torchvision import transforms
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter


from utils.gaze_utils import draw_gaze, angular_error, AverageMeter
from utils.args import str2bool
from utils.image_transform import my_normalize, my_denormalize, recover_image, create_grid

from datetime import datetime

def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Trainer(nn.Module):

    def __init__(self, model, train_loader, test_loader, output_dir=None, augment=False, lambda_values=None):
        super().__init__()

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model
        # self.causal_model = model
        self.augment = augment
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)
        # self.causal_model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.scheduler = StepLR(self.optimizer, step_size=5, gamma=0.1)

        self.start_epoch = 0
        self.epochs = 15
        self.train_iter = 0
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.ckpt_dir = osp.join(self.output_dir, 'ckpt')
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.tensorboard_dir = osp.join(self.output_dir, 'tensorboard')
        os.makedirs(self.tensorboard_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.tensorboard_dir)

        # Set lambda values with default fallback
        self.lambda_values = {
            'cf': 0.0,      # Counterfactual loss weight
            'cf_bg': 0.0,   # Background removal loss weight
            'dc': 0.0        # Weighted loss weight
        }
        
        # Override default lambda values if provided
        if lambda_values:
            for key, value in lambda_values.items():
                if key in self.lambda_values:
                    self.lambda_values[key] = value
        
        # Print lambda values
        print("Lambda Values:")
        for key, value in self.lambda_values.items():
            print(f"{key}: {value}")

    def train(self):
        for epoch in range(self.start_epoch, self.epochs):
            self.train_one_epoch(epoch)
            error = self.test(epoch)
            

            if epoch + 1 == 15:  # Saves only at the last epoch when epochs is 15
                add_file_name = 'epoch_' + str(epoch+1).zfill(2) + '_error=' + str(round(error, 2))
                self.save_checkpoint(
                    {'epoch': epoch + 1,
                    'model_state': self.model.module.state_dict() if isinstance(self.model, torch.nn.DataParallel) else self.model.state_dict(),
                    'optim_state': self.optimizer.state_dict(),
                    'schedule_state': self.scheduler.state_dict()
                    }, add=add_file_name
                )
        

    def one_iteration(self, data, tag='train'):
        l1_criterion = nn.L1Loss()
        input_var = data['image'].float().to(self.device)
        gaze_var = data['gaze'].float().to(self.device)
        input_var_zero = torch.zeros_like(input_var)
        input_var_rf = data.get('bg_image', torch.zeros_like(input_var)).float().to(self.device)
        # Forward pass for main model
        output_dict, output_dict_zero, output_dict_rf = self.model(input_var, input_var_zero, input_var_rf)
        pred_gaze = output_dict['pred_gaze']
        pred_gaze_zero = output_dict_zero['pred_gaze']
        pred_gaze_rf = output_dict_rf['pred_gaze']

        # Standard L1 Loss
        loss_gaze = l1_criterion(pred_gaze, gaze_var)
        zero_input_loss = l1_criterion(pred_gaze_zero, gaze_var)
        rf_input_loss = l1_criterion(pred_gaze_rf, gaze_var)
        
        # Weighted Loss based on eye height
        eye_height = data.get('average_eye_height', None).float().to(self.device)

        weights = 1 / (eye_height + 1e-6)  # Avoid division by zero
        weights = weights / weights.sum()  
        weighted_loss_scalar = (loss_gaze * weights).mean()
        
        # Combine Losses
        total_loss = (
            loss_gaze + 
            self.lambda_values['cf'] * zero_input_loss + 
            self.lambda_values['cf_bg'] * rf_input_loss + 
            self.lambda_values['dc'] * weighted_loss_scalar
        )
        
        # Calculate error
        error_gaze = np.mean(angular_error(pred_gaze.cpu().data.numpy(), gaze_var.cpu().data.numpy()))

        # Tensorboard logging
        if self.train_iter!=0 and self.train_iter % 10 == 0:
            self.writer.add_scalar(f'{tag}/loss_gaze', loss_gaze.item(), self.train_iter)
            self.writer.add_scalar(f'{tag}/total_loss', total_loss.item(), self.train_iter)
            self.writer.add_scalar(f'{tag}/error_gaze', error_gaze.item(), self.train_iter)
            log_img = torchvision.utils.make_grid(input_var[:8], nrow=4, normalize=True)   
            self.writer.add_image(f'{tag}/images', log_img, self.train_iter)
        
        self.train_iter += 1
        return total_loss, error_gaze, weighted_loss_scalar
    
    def train_one_epoch(self, epoch):
        print(f'Epoch: {epoch + 1} / {self.epochs}')
        self.model.train()
        for i, data in enumerate(track(self.train_loader, description='Training', transient=True)):
            total_loss, _, _ = self.one_iteration(data)
            
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

        self.scheduler.step()
        

    def test(self, epoch):
        errors_gaze = AverageMeter()
        total_losses = AverageMeter()

        self.model.eval()
        for i, data in enumerate(track(self.test_loader, description='Testing', transient=True)):
            with torch.no_grad():
                total_loss, error_gaze, _ = self.one_iteration(data, 'test')
                errors_gaze.update(error_gaze.item(), data['image'].size(0))
                total_losses.update(total_loss.item(), data['image'].size(0))

        print(f'Epoch: {epoch + 1}, Error gaze: {errors_gaze.avg}, Total Loss: {total_losses.avg}')
        self.model.train()

        self.writer.add_scalar('test/epoch_error_gaze', errors_gaze.avg, epoch + 1)
        self.writer.add_scalar('test/epoch_total_loss', total_losses.avg, epoch + 1)

        with open(osp.join(self.output_dir, 'test_results.txt'), 'a') as f:
            f.write('test on epoch {}, error: {}, total_loss: {}\n'.format(epoch + 1, errors_gaze.avg, total_losses.avg))
        
        return errors_gaze.avg

    def save_checkpoint(self, state, add=None):
        """
        Save a copy of the model
        """
        if add is not None:
            filename = add + '.pth.tar'
        else:
            filename = 'ckpt.pth.tar'
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        torch.save(state, ckpt_path)
        print('save file to: ', ckpt_path)

def build_model_from_cfg(cfg_path):
    cfg = OmegaConf.load(cfg_path)
    """
    cfg is like:f
        type: networks.GazeResNet.GazeRes18
        params: {}
    """
    module, cls = cfg['type'].rsplit(".", 1)
    MODEL = getattr(importlib.import_module(module, package=None), cls)
    model = MODEL(**cfg.get("params", dict()))
    return model
   

from utils.util import instantiate_from_config
if __name__ == '__main__':
    this_dir = os.path.dirname(os.path.realpath(__file__))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str,  help='name of the experiment')
    parser.add_argument('--model_cfg_path', help='the path to the config file of the model', default='/home/tpei0009/gaze-demo/gaze-demo/configs/models/causalres18.yaml')
    parser.add_argument('--output_dir', help='the path to the output directory', default=f'{this_dir}/logs_ours_average')
    
    # Add arguments for lambda values
    parser.add_argument('--lambda_cf', type=float, default=0.0, help='Counterfactual loss weight')
    parser.add_argument('--lambda_cf_bg', type=float, default=0.0, help='Background removal loss weight')
    parser.add_argument('--lambda_dc', type=float, default=0.5, help='Weighted loss weight')
    
    args = parser.parse_args()

    set_seed(42)
    
    output_dir = osp.join(args.output_dir, args.exp_name + '_ours')

    transform_mean_std = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}

    data_location = OmegaConf.load('./configs/data_path.yaml')
    data_cfg = {
        'xgaze': './configs/datasets/xgaze.yaml',
        'xgaze_full_light': './configs/datasets/xgaze_full_light.yaml',
        'xgaze_cam0': './configs/datasets/xgaze_cam0.yaml',
        'xgaze_full_light_cam0': './configs/datasets/xgaze_full_light_cam0.yaml',
        'mpii': './configs/datasets/mpii.yaml',
        'gazecapture_test': './configs/datasets/gazecapture.yaml',
        'xgaze_gender_balance': '/home/tpei0009/gaze-demo/gaze-demo/configs/datasets/xgaze_gender_balance.yaml',
        'xgaze_eye_height_balance': '/home/tpei0009/gaze-demo/gaze-demo/configs/datasets/xgaze_eye_height_balance.yaml',
        'mpii_nv': '/home/tpei0009/gaze-demo/gaze-demo/configs/datasets/mpii_nv.yaml',
        'xgaze_60':'./configs/datasets/xgaze_60.yaml',
        'xgaze_20':'./configs/datasets/xgaze_20.yaml',
        'gaze360_train': '/home/tpei0009/gaze-demo/gaze-demo/configs/datasets/gaze360.yaml'
    }

    exps = {
        'gaze360_to_mpii':['gaze360_train', 'mpii'],
        'xgaze_to_mpii': ['xgaze','mpii'],
        'xgaze_to_gazecapture': ['xgaze','gazecapture_test'], 
        'mpii_to_mpii': ['mpii','mpii'],
        'xgaze_to_xgaze': ['xgaze','xgaze'],
        }
    
    train_data_name = exps[args.exp_name][0]
    train_cfg = OmegaConf.load(data_cfg[train_data_name])
    train_cfg['params']['transform_Normalize'] = transform_mean_std
    train_cfg['params']['dataset_path'] = data_location['xgaze' if 'xgaze' in train_data_name else train_data_name]
    train_dataset = instantiate_from_config(train_cfg)

    test_data_name = exps[args.exp_name][1]
    test_cfg = OmegaConf.load(data_cfg[test_data_name])
    test_cfg['params']['transform_Normalize'] = transform_mean_std
    test_cfg['params']['dataset_path'] = data_location['xgaze' if 'xgaze' in test_data_name else test_data_name]
    test_datset = instantiate_from_config(test_cfg)
    
    print(f'train_dataset {train_data_name} num of samples: ', len(train_dataset))
    print(f'test_dataset {test_data_name} num of samples: ', len(test_datset))

    train_loader = DataLoader(train_dataset, batch_size=200, shuffle=True, num_workers=18)
    test_loader = DataLoader(test_datset, batch_size=200, shuffle=False, num_workers=18)
    
    model = build_model_from_cfg(args.model_cfg_path)
    # print(model)
    # summary(model)

    os.makedirs(output_dir, exist_ok=True)
    OmegaConf.save(transform_mean_std, osp.join(output_dir, 'transform_mean_std.yaml'))

    # Prepare lambda values dictionary
    lambda_values = {
        'cf': args.lambda_cf,
        'cf_bg': args.lambda_cf_bg,
        'dc': args.lambda_dc
    }

    trainer = Trainer(
        model,
        train_loader,
        test_loader,
        output_dir=output_dir, 
        lambda_values=lambda_values  # Pass lambda values to Trainer
    )
    trainer.train()