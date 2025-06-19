import os, sys
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import math
import copy
import torch.utils.model_zoo as model_zoo
from torch.utils.model_zoo import load_url as load_state_dict_from_url
import torch.nn.functional as F

from .modules.resnetpure import resnet50, resnet18, BasicBlock


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


class ResDeconv(nn.Module):

	def __init__(self, inplanes, block):
		self.inplanes=inplanes
		super(ResDeconv, self).__init__()
		model = []
		model += [nn.Upsample(scale_factor=2)]
		model += [self._make_layer(block, 256, 2)] # 28
		model += [nn.Upsample(scale_factor=2)]
		model += [self._make_layer(block, 128, 2)] # 56
		model += [nn.Upsample(scale_factor=2)]
		model += [self._make_layer(block, 64, 2)] # 112
		model += [nn.Upsample(scale_factor=2)]
		model += [self._make_layer(block, 32, 2)] # 112
		model += [nn.Upsample(scale_factor=2)]
		model += [self._make_layer(block, 16, 2)] # 112
		model += [nn.Conv2d(16, 3, stride=1, kernel_size=1)]

		self.deconv = nn.Sequential(*model)


		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def _make_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				nn.Conv2d(self.inplanes, planes * block.expansion,
						  kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(planes * block.expansion),
			)

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes))

		return nn.Sequential(*layers)

	def forward(self, features):
		img = self.deconv(features)
		return img


class ResGazeEs(nn.Module):
	def __init__(self, inplanes):
		super().__init__()
		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

		self.projector = nn.Sequential(
			nn.Linear(inplanes, inplanes),
			nn.ReLU(inplace=True),
		)
		self.fc = nn.Linear(inplanes, 2)


	def forward(self, features):
		x = self.avgpool(features)
		x = x.view(x.size(0), -1)
		pred_gaze = self.fc( self.projector(x) )
		return pred_gaze
	
	


class PureGaze_18(nn.Module):
	def __init__(self):
		super().__init__()

		self.feature =  resnet18(pretrained=True)
		self.gazeEs = ResGazeEs(inplanes=512)
		self.deconv = ResDeconv(inplanes=512, block=BasicBlock)
		self.out_activation = nn.Tanh()
	
	def forward(self, imgs, decode=False ):
		imgs = re_normalize(imgs, old='imagenet', new='[-1,1]')
		output_dict = {}
		batch_size = imgs.size(0)
		features = self.feature(imgs) ## (b, 3, h, w) --> (b, 2048 (res50) or 512 (res18), n, m)
		pred_gaze = self.gazeEs(features)
		output_dict["pred_gaze"] = pred_gaze
		if decode:
			img = self.deconv(features)
			img = self.out_activation(img)
			img = re_normalize(img, old='[-1,1]', new='imagenet')
		else:
			img = None
		output_dict["pred_img"] = img
		return output_dict
	
class PureGaze_50(nn.Module):
	def __init__(self):
		super().__init__()

		self.feature =  resnet50(pretrained=True)
		self.gazeEs = ResGazeEs(inplanes=2048)
		self.deconv = ResDeconv(inplanes=2048, block=BasicBlock)
		self.out_activation = nn.Tanh()

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
	def forward(self, imgs, decode=False ):
		imgs = re_normalize(imgs, old='imagenet', new='[-1,1]')
		output_dict = {}
		batch_size = imgs.size(0)
		features = self.feature(imgs) ## (b, 3, h, w) --> (b, 2048 (res50) or 512 (res18), n, m)
		pred_gaze = self.gazeEs(features)
		output_dict["pred_gaze"] = pred_gaze
		if decode:
			img = self.deconv(features)
			img = self.out_activation(img)
			img = re_normalize(img, old='[-1,1]', new='imagenet')
		else:
			img = None
		output_dict["pred_img"] = img
		return output_dict




