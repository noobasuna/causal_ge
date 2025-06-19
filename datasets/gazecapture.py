import os, json, yaml, random
import os.path as osp
from glob import glob
import numpy as np
import h5py
import json
import cv2
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset


from typing import List
from omegaconf import OmegaConf

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
# MEAN = [0.5,0.5,0.5]
# STD = [0.5,0.5,0.5]


class GazeCaptureDataset(Dataset):
	def __init__(self, 
				dataset_path: str, 
				color_type,
				keys_to_use: List[str] = None, 
				data_name=None, 
				image_size:int=224,
				image_key='face_patch',
				gaze_key='face_gaze',
				transform_Normalize=None, 
				):
		self.path = dataset_path
		self.hdfs = {}
		self.data_name = data_name
		self.image_key = image_key
		self.gaze_key = gaze_key

		self.image_size = (image_size, image_size)

		assert color_type in ['rgb', 'bgr']
		self.color_type = color_type
		self.selected_keys = [ k + '.h5' for k in keys_to_use]
		assert len(self.selected_keys) > 0
		
		self.file_paths = [os.path.join(self.path, k) for k in self.selected_keys]
		for num_i in range(0, len(self.selected_keys)):
			file_path = os.path.join(self.path, self.selected_keys[num_i]) # the subdirectories: train, test are not used in MPIIFaceGaze and MPII_Rotate
			self.hdfs[num_i] = h5py.File(file_path, 'r', swmr=True)
			print('read file: ', os.path.join(self.path, self.selected_keys[num_i]))
			assert self.hdfs[num_i].swmr_mode

		self.idx_to_kv = []
		for num_i in range(0, len(self.selected_keys)):
			n = self.hdfs[num_i][self.image_key].shape[0] 
			self.idx_to_kv += [(num_i, i) for i in range(n)]

		for num_i in range(0, len(self.hdfs)):            
			if self.hdfs[num_i]:
				self.hdfs[num_i].close()
				self.hdfs[num_i] = None

		self.__hdfs = None
		self.hdf = None
		self.transform = transforms.Compose([
			transforms.ToPILImage(),
			transforms.ToTensor(),  # this also convert pixel value from [0,255] to [0,1]
			transforms.Normalize(mean=transform_Normalize['mean'], std=transform_Normalize['std']) ,
		])

	def __len__(self):
		return len(self.idx_to_kv)

	def __del__(self):
		for num_i in range(0, len(self.hdfs)):
			if self.hdfs[num_i]:
				self.hdfs[num_i].close()
				self.hdfs[num_i] = None
		
	@property
	def archives(self):
		if self.__hdfs is None: # lazy loading here!
			self.__hdfs = [h5py.File(h5_path, "r", swmr=True) for h5_path in self.file_paths]
		return self.__hdfs


	def preprocess_image(self, image):
		image = image.astype(np.float32)
		if self.color_type == 'bgr':
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_AREA)
		image = self.transform(image.astype(np.uint8)		)
		return image

	def __getitem__(self, index):
		
		key, idx = self.idx_to_kv[index]
		self.hdf = self.archives[key]
		assert self.hdf.swmr_mode
		image = self.hdf[self.image_key][idx, :]
		gaze_label = self.hdf[self.gaze_key][idx].astype('float') if self.gaze_key in self.hdf else np.array([0,0]).astype('float')
		head_label = self.hdf['face_head_pose'][idx].astype('float') if 'face_head_pose' in self.hdf else np.array([0,0]).astype('float')

		entry = {
			'image': self.preprocess_image(image),
			'gaze': gaze_label,
			'head': head_label,
			'key': key,
			'index':index
		}
	
		return entry
