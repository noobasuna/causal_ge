# ---------------------- general -------------------
from collections import OrderedDict
from re import S

from networkx import sigma
import imageio
import math
import os
import sys
import numpy as np
import yaml
import cv2
import shutil
import time
import importlib
import copy
from tqdm import tqdm
from rich.progress import track
from glob import glob
from omegaconf import OmegaConf
import random
# ----------------------- torch ------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as tvF
from torch.autograd import Variable
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset, Sampler

from main import set_seed
from utils import instantiate_from_config
from utils.util import call_model_method, get_attributes_with_prefix, update_ema_params
from utils.gaze import angular_error
from utils.image_transform import recover_image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(font_scale=1.5)
font = {'family' : 'sans-serif',
		'serif': 'Helvetica'}
matplotlib.rc('font', **font)




import cv2
class Gelossop():
	def __init__(self, attentionmap, w1=3, w2=1):
		# print(" attentionmap: ", attentionmap)
		attentionmap = cv2.imread(attentionmap, 0)/255
		attentionmap = torch.from_numpy(attentionmap).type(torch.FloatTensor)
		self.gloss = torch.nn.L1Loss().cuda()
		#self.gloss = torch.nn.MSELoss().cuda()
		self.recloss = torch.nn.MSELoss().cuda()
		self.attentionmap = attentionmap.cuda()
		self.w1 = w1
		self.w2 = w2
		

	def __call__(self, gaze, img, gaze_pre, img_pre, compute_loss1=True):
		# loss2 = 1-self.recloss(img, img_pre)
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
		self.recloss = torch.nn.MSELoss().cuda()

	def __call__(self, img, img_pre):
		return self.recloss(img, img_pre)




class PureGaze_SimpleTrainer(Baseline_SimpleTrainer):
	def configure_loss(self, config):
		self.gaze_loss = torch.nn.L1Loss()
		self.geloss_op = instantiate_from_config(config.geloss_op_config)
		self.deloss_op = instantiate_from_config(config.deloss_op_config)
	

	def configure_optim_scheduler(self, config):
		lr = 0.0001
		self.optimizer = optim.Adam(self.model.feature.parameters(), lr=lr, betas=(0.9,0.95)) ## dummy optimizer
		self.scheduler = StepLR(self.optimizer, step_size=10, gamma=0.1) ## dummy scheduler
		self.ge_optimizer = optim.Adam(self.model.feature.parameters(), lr=lr, betas=(0.9,0.95))

		self.ga_optimizer = optim.Adam(self.model.gazeEs.parameters(), lr=lr, betas=(0.9,0.95))

		self.de_optimizer = optim.Adam(self.model.deconv.parameters(), lr=lr, betas=(0.9,0.95))
		

	def optimize(self, input_var, gaze_var, head_var, losses_dict):
		out_dict = self.model(input_var, normalize_z=self.normalize_z, decode=True)

		pred_gaze = out_dict['pred_gaze']
		pred_img = out_dict['pred_img']

		self.ge_optimizer.zero_grad()
		self.ga_optimizer.zero_grad()
		self.de_optimizer.zero_grad()
		if isinstance(self.model, torch.nn.DataParallel):
			for param in self.model.module.feature.parameters():
				param.requires_grad = False
		else:
			for param in self.model.deconv.parameters():
				param.requires_grad=False

		# loss calculation
		geloss = self.geloss_op(pred_gaze, pred_img, gaze_var, input_var)
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

		# for i, entry in enumerate(tqdm(self.source_loader)):
		source_iter = iter(self.source_loader)
		num_iters = len(self.source_loader)

		for i in tqdm(range( num_iters )):
			try: 
				entry = next(source_iter)
			except StopIteration: # restart the generator if the previous generator is exhausted.
				source_iter = iter(self.source_loader)
				entry = next(source_iter)
			vis_entry = {}
			loss_all = 0
			##### 1. source data ----------------------------------
			input_var = entry['image'].float().cuda()
			gaze_var = entry['gaze'].float().cuda()
			head_var = entry['head'].float().cuda()
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
		if i % ( len(self.source_loader) // 10 ) != 0:
			return
		vis_dir = os.path.join(self.image_dir, tag)
		os.makedirs(vis_dir, exist_ok=True)
		first_key = list(vis_entry.keys())[0]
		batch_size = vis_entry[first_key][0].size(0)

		vis_entry = [ (recover_image(image_tensor), label_tensor.cpu().data.numpy(), pred_tensor.cpu().data.numpy()) 
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
		print("\n[*] Train on {} samples (source)".format(len(self.source_loader.dataset)))
		self.outfile = open(os.path.join(self.logging_dir, "train_log"), 'w') 


		for epoch in range(self.start_epoch, self.epochs):
			print('\nEpoch: {}/{}'.format(epoch + 1, self.epochs))
			self.model.train()
			self.base_epoch(epoch)
			if (epoch+1) % self.config.save_epoch == 0 or epoch == self.epochs - 1:
				add_file_name = 'epoch_' + str(epoch+1).zfill(2)
				self.save_checkpoint(
					{'epoch': epoch + 1,
					'model_state': self.model.module.state_dict() if isinstance(self.model, torch.nn.DataParallel) else self.model.state_dict(),
					'optim_state': self.optimizer.state_dict(),
					'schedule_state': self.scheduler.state_dict()
					}, add=add_file_name 
				)
			
			if (epoch+1) % self.config.valid_epoch == 0 or epoch == self.epochs - 1:
				print_cyan('validate at epoch: ', epoch)
				self.test(epoch, eval_tag='val_dataloader')
				self.test(epoch, eval_tag='val_dataloader_2')
				self.test(epoch, eval_tag='val_dataloader_3')
				self.test(epoch, eval_tag='val_dataloader_4')
				
				if (epoch+1) % self.config.eval_epoch == 0 or epoch == self.epochs - 1:
					print_cyan('test at epoch: ', epoch)
					self.test(epoch, eval_tag='test_dataloader')
					self.test(epoch, eval_tag='test_dataloader_2')
					self.test(epoch, eval_tag='test_dataloader_3')
					self.test(epoch, eval_tag='test_dataloader_4')
					self.test(epoch, eval_tag='test_dataloader_5')
