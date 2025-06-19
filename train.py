
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

    def __init__(self, model, train_loader, test_loader, output_dir=None, augment=False):
        super().__init__()

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model
        self.augment = augment
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)

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

    def train(self):
        for epoch in range(self.start_epoch, self.epochs):
            self.train_one_epoch(epoch)
            error = self.test(epoch)
            

            if (epoch + 1) % max( 1, (self.epochs//3) ) == 0:
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
        
        output_dict = self.model(input_var)
        pred_gaze = output_dict['pred_gaze']
        loss_gaze = l1_criterion(pred_gaze, gaze_var)
        error_gaze = np.mean(angular_error(pred_gaze.cpu().data.numpy(), gaze_var.cpu().data.numpy()))

        if self.train_iter!=0 and self.train_iter % 10 == 0:
            self.writer.add_scalar( f'{tag}/loss_gaze', loss_gaze.item(), self.train_iter)
            self.writer.add_scalar( f'{tag}/error_gaze', error_gaze.item(), self.train_iter)
            log_img = torchvision.utils.make_grid(input_var[:8], nrow=4, normalize=True)   
            self.writer.add_image( f'{tag}/images', log_img, self.train_iter)


        # if self.augment and 'augmented_image' in data.keys():
        #     aug_input_var = data['augmented_image'].float().to(self.device)
        #     aug_pred_gaze = self.model(aug_input_var)
        #     aug_loss_gaze = l1_criterion(aug_pred_gaze, gaze_var)
        #     aug_error_gaze = np.mean(angular_error(aug_pred_gaze.cpu().data.numpy(), gaze_var.cpu().data.numpy()))
            
        #     loss_gaze += aug_loss_gaze

        #     if self.train_iter!=0 and self.train_iter % 10 == 0:
        #         self.writer.add_scalar( f'{tag}/aug_loss_gaze', aug_loss_gaze.item(), self.train_iter)
        #         self.writer.add_scalar( f'{tag}/aug_error_gaze', aug_error_gaze.item(), self.train_iter)
        #         log_img_aug = torchvision.utils.make_grid(aug_input_var[:8], nrow=4, normalize=True)
        #         self.writer.add_image(f'{tag}/augmented_images', log_img_aug, self.train_iter)

        self.train_iter += 1
        return loss_gaze, error_gaze
    
    def train_one_epoch(self, epoch):
        print(f'Epoch: {epoch + 1} / {self.epochs}')
        self.model.train()
        for i, data in enumerate(track(self.train_loader, description='Training', transient=True)):
            
            loss, _ = self.one_iteration(data)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.scheduler.step()
        

    def test(self, epoch):
        errors_gaze = AverageMeter()

        self.model.eval()
        for i, data in enumerate(track(self.test_loader, description='Testing', transient=True)):
            _, error_gaze = self.one_iteration(data, 'test')
            errors_gaze.update(error_gaze.item(),  data['image'].size(0))

        print(f'Epoch: {epoch + 1}, Error gaze: {errors_gaze.avg}')
        self.model.train()

        self.writer.add_scalar('test/epoch_error_gaze', errors_gaze.avg, epoch + 1)

        with open(osp.join(self.output_dir, 'test_results.txt'), 'a') as f:
            f.write('test on epoch {}, error: {}\n'.format(epoch + 1, errors_gaze.avg))
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
    cfg is like:
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
    parser.add_argument('--model_cfg_path', help='the path to the config file of the model', default=f'{this_dir}/configs/models/res18.yaml')
    parser.add_argument('--output_dir', help='the path to the output directory', default=f'{this_dir}/logs')
    args = parser.parse_args()

    set_seed(42)
    
    output_dir = osp.join(args.output_dir, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    transform_mean_std = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}

    data_location = OmegaConf.load('./configs/data_path.yaml')
    data_cfg = {
        'xgaze': './configs/datasets/xgaze.yaml',
        'xgaze_full_light': './configs/datasets/xgaze_full_light.yaml',
        'xgaze_cam0': './configs/datasets/xgaze_cam0.yaml',
        'xgaze_full_light_cam0': './configs/datasets/xgaze_full_light_cam0.yaml',
        'mpii': './configs/datasets/mpii.yaml',
        'gazecapture_test': './configs/datasets/gazecapture.yaml',
        'gaze360_train': './configs/datasets/gaze360.yaml',
        'gaze360_test': './configs/datasets/gaze360_test.yaml',
    }

    exps = {
        'xgaze_to_xgaze_cam0': ['xgaze', 'xgaze_cam0'],
        'xgaze_cam0_to_mpii': ['xgaze_cam0', 'mpii'],
        'xgaze_cam0_to_cam0': ['xgaze_cam0', 'xgaze_cam0'],
        'xgaze_to_xgaze_cam0': ['xgaze', 'xgaze_cam0'],
        'full_light_xgaze_to_xgaze_cam0': ['xgaze_full_light', 'xgaze_full_light_cam0'],
        'full_light_xgaze_cam0_to_mpii': ['xgaze_full_light_cam0', 'mpii'],
        'gazecapture_to_xgaze_full_light_cam0': ['gazecapture_test', 'xgaze_full_light_cam0'],
        'gazecapture_to_mpii': ['gazecapture', 'mpii'],
        'gaze360_train_to_gaze360_test': ['gaze360_train', 'gaze360_test'],
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

    train_loader = DataLoader(train_dataset, batch_size=200, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_datset, batch_size=200, shuffle=False, num_workers=8)
    
    model = build_model_from_cfg(args.model_cfg_path)
    # print(model)
    # summary(model)

    os.makedirs(output_dir, exist_ok=True)
    OmegaConf.save(transform_mean_std, osp.join(output_dir, 'transform_mean_std.yaml'))
    trainer = Trainer(
        model,
        train_loader,
        test_loader,
        output_dir=output_dir, 
    )
    trainer.train()
   

