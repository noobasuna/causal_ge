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
from collections import OrderedDict

import torchvision
from torchvision import transforms
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter


from utils.gaze_utils import draw_gaze, angular_error, AverageMeter
from utils.args import str2bool
from utils.image_transform import my_normalize, my_denormalize, recover_image, create_grid

from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(font_scale=1.5)
font = {'family' : 'sans-serif',
		'serif': 'Helvetica'}
matplotlib.rc('font', **font)

def set_seed(seed_value=42):
	random.seed(seed_value)
	np.random.seed(seed_value)
	torch.manual_seed(seed_value)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed_value)
		torch.cuda.manual_seed_all(seed_value)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
		
def re_normalize(image_tensor, old='[-1,1]', new='imagenet'):
		"""
		Re-normalizes an image tensor from one normalization scheme to another.
		Args:
			image_tensor (torch.Tensor): Image tensor to be re-normalized.
			old (str): Old normalization scheme. Options: '[-1,1]', 'imagenet'.
			new (str): New normalization scheme. Options: '[-1,1]', 'imagenet'.
		Returns:
			torch.Tensor: Re-normalized image tensor.
		"""
		# Old normalization parameters
		device = image_tensor.device
		if old == '[-1,1]':
			old_mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(device)
			old_std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(device)
		elif old == 'imagenet':
			old_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
			old_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
		elif old == '[0,1]':
			old_mean = torch.tensor([0.0, 0.0, 0.0]).view(1, 3, 1, 1).to(device)
			old_std = torch.tensor([1.0, 1.0, 1.0]).view(1, 3, 1, 1).to(device)
		else:
			print('old normalization not implemented')
			raise NotImplementedError
		# New normalization parameters
		if new == '[-1,1]':
			new_mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(device)
			new_std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(device)
		elif new == 'imagenet':
			new_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
			new_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
		elif new == '[0,1]':
			new_mean = torch.tensor([0.0, 0.0, 0.0]).view(1, 3, 1, 1).to(device)
			new_std = torch.tensor([1.0, 1.0, 1.0]).view(1, 3, 1, 1).to(device)
		else:
			print('new normalization not implemented')
			raise NotImplementedError
		# Step 1: Denormalize the image tensor using the old mean and std
		denormalized_image = image_tensor * old_std + old_mean
		# Step 2: Normalize the image tensor using the new mean and std
		normalized_image = (denormalized_image - new_mean) / new_std

		return normalized_image

class Trainer(nn.Module):

	def __init__(self, model, train_loader, test_loader, output_dir=None, augment=False):
		super().__init__()

		self.train_loader = train_loader
		self.test_loader = test_loader
		self.model = model
		self.augment = augment
		# Use CPU if CUDA is not available or if user forces CPU
		self.device = torch.device("cpu")  # Force CPU usage

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
	cfg is like:f
		type: networks.GazeResNet.GazeRes18
		params: {}
	"""
	module, cls = cfg['type'].rsplit(".", 1)
	MODEL = getattr(importlib.import_module(module, package=None), cls)
	model = MODEL(**cfg.get("params", dict()))
	return model
   
class Gelossop():
	def __init__(self, w1=3, w2=1):
		# print(" attentionmap: ", attentionmap)
		attentionmap = cv2.imread("/home/tpei0009/gaze-demo/gaze-demo/PureGaze/Masker/eth-masker.jpg", 0)/255
		attentionmap = torch.from_numpy(attentionmap).type(torch.FloatTensor)
		self.gloss = torch.nn.L1Loss()  # Remove .cuda()
		self.recloss = torch.nn.MSELoss()  # Remove .cuda()
		self.attentionmap = attentionmap  # Do not move to cuda
		self.w1 = w1
		self.w2 = w2
		

	def __call__(self, gaze, img, gaze_pre, img_pre, compute_loss1=True):
		# loss2 = 1-self.recloss(img, img_pre)
		# print('img shape: ', img.shape, 'img_pre shape: ', img_pre.shape)
		loss2 = 1 - (img - img_pre)**2
		zeros = torch.zeros_like(loss2)
		loss2 = torch.where(loss2 < 0.75, zeros, loss2)
		attentionmap = F.interpolate(self.attentionmap.unsqueeze(0).unsqueeze(0), size=(loss2.size(2), loss2.size(3)), mode='bilinear', align_corners=True)
		loss2 = torch.mean(attentionmap * loss2)
		loss_all = self.w2 * loss2
		if compute_loss1:
			loss1 = self.gloss(gaze, gaze_pre)
			loss_all += self.w1 * loss1
		return loss_all

class Delossop():
	def __init__(self):
		self.recloss = torch.nn.MSELoss()  # Remove .cuda()

	def __call__(self, img, img_pre):
		return self.recloss(img, img_pre)


class PureGaze_SimpleTrainer(Trainer):
	def __init__(self, model, train_loader, test_loader, output_dir=None, augment=False, config=None):
		super().__init__(model, train_loader, test_loader, output_dir, augment)
		self.configure_loss(config)  # Ensure loss configurations
		self.configure_optim_scheduler(config)  # Initialize optimizers
		
	def configure_loss(self, config):
		self.gaze_loss = torch.nn.L1Loss()
		self.geloss_op = Gelossop(w1=1, w2=1)
		self.deloss_op = Delossop()

	def configure_optim_scheduler(self, config):
		lr = 0.0001
		self.optimizer = optim.Adam(self.model.feature.parameters(), lr=lr, betas=(0.9,0.95)) ## dummy optimizer
		self.scheduler = StepLR(self.optimizer, step_size=10, gamma=0.1) ## dummy scheduler
		self.ge_optimizer = optim.Adam(self.model.feature.parameters(), lr=lr, betas=(0.9,0.95))

		self.ga_optimizer = optim.Adam(self.model.gazeEs.parameters(), lr=lr, betas=(0.9,0.95))

		self.de_optimizer = optim.Adam(self.model.deconv.parameters(), lr=lr, betas=(0.9,0.95))

	def optimize(self, input_var, gaze_var, head_var, losses_dict):
		out_dict = self.model(input_var, decode=True)#, normalize_z=self.normalize_z

		pred_gaze = out_dict['pred_gaze']
		pred_img = out_dict['pred_img']

		self.ge_optimizer.zero_grad()
		self.ga_optimizer.zero_grad()
		self.de_optimizer.zero_grad()
		if isinstance(self.model, torch.nn.DataParallel):
			for param in self.model.module.deconv.parameters():
					param.requires_grad = False
			else:
				for param in self.model.deconv.parameters():
					param.requires_grad = False

		# loss calculation
		geloss = self.geloss_op(gaze_var, input_var, pred_gaze, pred_img)
		geloss.backward(retain_graph=True)
		losses_dict['geloss'] = geloss.item()

		if isinstance(self.model, torch.nn.DataParallel):
			for param in self.model.module.deconv.parameters():
				param.requires_grad = True
			for param in self.model.module.feature.parameters():
				param.requires_grad = False
		else:
			for param in self.model.deconv.parameters():
				param.requires_grad=True
			for param in self.model.feature.parameters():
				param.requires_grad = False

		deloss = self.deloss_op(pred_img, input_var)
		deloss.backward()
		losses_dict['deloss'] = deloss.item()

		if isinstance(self.model, torch.nn.DataParallel):
			for param in self.model.module.feature.parameters():
				param.requires_grad = True
		else:
			for param in self.model.feature.parameters():
				param.requires_grad=True

		self.ge_optimizer.step()
		self.ga_optimizer.step()
		self.de_optimizer.step()
		return out_dict


	def base_epoch(self, epoch):
		"""
			base means only gaze estimation loss
		"""
		self.model.train()
		losses_dict = OrderedDict()
		errors_dict = OrderedDict()
		self.metrics = { }

		# for i, entry in enumerate(tqdm(self.train_loader)):
		train_iter = iter(self.train_loader)
		num_iters = len(self.train_loader)

		for i in tqdm(range( num_iters )):
			try: 
				entry = next(train_iter)
			except StopIteration:
				train_iter = iter(self.train_loader)
				entry = next(train_iter)
			vis_entry = {}
			loss_all = 0
			##### 1. source data ----------------------------------
			input_var = entry['image'].float().to(self.device)
			gaze_var = entry['gaze'].float().to(self.device)
			head_var = entry['head'].float().to(self.device)
			batch_size = input_var.size(0)

			out_dict = self.optimize(input_var, gaze_var, head_var, losses_dict)
			pred_gaze = out_dict['pred_gaze']
			vis_entry['input'] = (input_var, gaze_var, pred_gaze)
			pred_img = out_dict['pred_img']
			vis_entry['pred_img'] = (pred_img, gaze_var, gaze_var)

			errors_dict['error_s_gaze'] = np.mean(angular_error(pred_gaze.cpu().data.numpy(), gaze_var.cpu().data.numpy()))
		
			self.print_entry(i, vis_entry,'vis_train')

			self.train_iter = self.train_iter + 1
			
	
	def print_entry(self, i, vis_entry, tag):
		if i % ( len(self.train_loader) // 10 ) != 0:
			return
		vis_dir = os.path.join(self.output_dir, tag)
		os.makedirs(vis_dir, exist_ok=True)
		first_key = list(vis_entry.keys())[0]
		
		batch_size = vis_entry[first_key][0].size(0)

		# Use ImageNet normalization parameters
		MEAN = [0.485, 0.456, 0.406]
		STD = [0.229, 0.224, 0.225]
		
		vis_entry = [ (recover_image(image_tensor, MEAN, STD), label_tensor, pred_tensor) 
					for entry_type, (image_tensor, label_tensor, pred_tensor) in vis_entry.items() ]

		for b in range(min(batch_size, 2) ):
			img_vis = []
			for idx, (img, label, pred) in enumerate(vis_entry):
				img_b = img[b]
				label_b = label[b]
				pred_b = pred[b]
				# img_b = draw_gaze(img_b, label_b, color=(0, 255, 0))
				# img_b = draw_gaze(img_b, pred_b, color=(0, 0, 255))
				img_vis.append(img_b)
			img_vis = cv2.hconcat(img_vis)
			cv2.imwrite(os.path.join(vis_dir, 'batch_{}_{}.jpg'.format(self.train_iter, b)), img_vis)

	def train(self):
		# print_cyan("start training")
		print("\n[*] Train on {} samples (source)".format(len(self.train_loader.dataset)))
		self.outfile = open(os.path.join(self.output_dir, "train_log"), 'w') 


		for epoch in range(self.start_epoch, self.epochs):
			print('\nEpoch: {}/{}'.format(epoch + 1, self.epochs))
			self.model.train()
			self.base_epoch(epoch)
			if (epoch+1) % 5 == 0 or epoch == self.epochs - 1:
				add_file_name = 'epoch_' + str(epoch+1).zfill(2)
				self.save_checkpoint(
					{'epoch': epoch + 1,
					'model_state': self.model.module.state_dict() if isinstance(self.model, torch.nn.DataParallel) else self.model.state_dict(),
					'optim_state': self.optimizer.state_dict(),
					'schedule_state': self.scheduler.state_dict()
					}, add=add_file_name 
				)
			
			if (epoch+1) % 1 == 0 or epoch == self.epochs - 1:
				print('validate at epoch: ', epoch)
				self.test(epoch)#, eval_tag='val_dataloader')
				# self.test(epoch, eval_tag='val_dataloader_2')
				# self.test(epoch, eval_tag='val_dataloader_3')
				# self.test(epoch, eval_tag='val_dataloader_4')
				
				if (epoch+1) % 1 == 0 or epoch == self.epochs - 1:
					print('test at epoch: ', epoch)
					self.test(epoch)#, eval_tag='test_dataloader')
					# self.test(epoch, eval_tag='test_dataloader_2')
					# self.test(epoch, eval_tag='test_dataloader_3')
					# self.test(epoch, eval_tag='test_dataloader_4')
					# self.test(epoch, eval_tag='test_dataloader_5')

from utils.util import instantiate_from_config
if __name__ == '__main__':
	this_dir = os.path.dirname(os.path.realpath(__file__))
	# Force CPU usage
	device = torch.device("cpu")

	parser = argparse.ArgumentParser()
	parser.add_argument('--exp_name', type=str,  help='name of the experiment')
	parser.add_argument('--model_cfg_path', help='the path to the config file of the model', default='/home/tpei0009/gaze-demo/gaze-demo/configs/models/puregaze.yaml')
	parser.add_argument('--output_dir', help='the path to the output directory', default=f'{this_dir}/logs')
	args = parser.parse_args()

	set_seed(42)
	
	output_dir = osp.join(args.output_dir, args.exp_name + '_puregaze')

	transform_mean_std = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}

	data_location = OmegaConf.load('./configs/data_path.yaml')
	data_cfg = {
		'xgaze': './configs/datasets/xgaze.yaml',
		'mpii': './configs/datasets/mpii.yaml',
		'gazecapture_test': './configs/datasets/gazecapture.yaml',
		'xgaze_full_light_cam0': './configs/datasets/xgaze_full_light_cam0.yaml',
		'gaze360_train':'./configs/datasets/gaze360.yaml',
		'gaze360_test':'./configs/datasets/gaze360_test.yaml',
		'eyediap_cs': '/home/tpei0009/gaze-demo/gaze-demo/configs/datasets/eyediap_cs.yaml',
		'eyediap_ft': '/home/tpei0009/gaze-demo/gaze-demo/configs/datasets/eyediap_ft.yaml'
	}

	exps = {
		'xgaze_to_mpii': ['xgaze','mpii'],
		'xgaze_to_diap_cs': ['xgaze','eyediap_cs'],
		'xgaze_to_diap_ft': ['xgaze','eyediap_ft'],
		'gaze360_to_diap_cs': ['gaze360_train','eyediap_cs'],
		'gaze360_to_diap_ft': ['gaze360_train','eyediap_ft'],
		'xgaze_to_gazecapture': ['xgaze','gazecapture_test'], 
		'gaze360_to_mpii':['gaze360_train', 'mpii'],
		'mpii_to_mpii': ['mpii','mpii'],
		'xgaze_full_light_cam0_to_mpii':['xgaze_full_light_cam0', 'mpii'],
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

	train_loader = DataLoader(train_dataset, batch_size=200, shuffle=True, num_workers=12)
	test_loader = DataLoader(test_datset, batch_size=200, shuffle=False, num_workers=12)
	
	model = build_model_from_cfg(args.model_cfg_path)
	print(model)
	# summary(model)

	os.makedirs(output_dir, exist_ok=True)
	OmegaConf.save(transform_mean_std, osp.join(output_dir, 'transform_mean_std.yaml'))
	trainer = PureGaze_SimpleTrainer(
		model,
		train_loader,
		test_loader,
		output_dir=output_dir, 
	)
	trainer.train()