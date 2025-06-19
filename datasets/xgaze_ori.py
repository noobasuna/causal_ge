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

import mediapipe as mp
import math

# Initialize MediaPipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

class XGazeDataset(Dataset):
	def __init__(self, 
				 dataset_path: str, 
				 color_type,
				 keys_to_use: List[str] = None, 
				 data_name=None, 
				 image_size: int = 224,
				 image_key='face_patch',
				 gaze_key='face_gaze',
				 eye_height_label='eye_height',
				 transform_Normalize={'mean': MEAN, 'std': STD}, 
				 camera_tag='all',
				 full_light_only=False,
				 ):
		self.path = dataset_path
		self.hdfs = {}

		self.image_key = image_key
		self.gaze_key = gaze_key
		self.eye_height_label= eye_height_label

		self.image_size = (image_size, image_size)

		assert color_type in ['rgb', 'bgr']
		self.color_type = color_type

		assert camera_tag in ['all', 'cam00', 'small_cam_2', 'small_cam_8', 'small_cam_2_c', 'small', 'cam_0127']
		self.camera_tag = camera_tag
		self.small_cam_8 = [0, 1, 2, 3, 6, 7, 8, 9]
		self.small_cam_8_c = [x for x in list(range(18)) if x not in self.small_cam_8]
		self.small_cam_2 = [0, 1]
		self.small_cam_2_c = [x for x in list(range(18)) if x not in self.small_cam_2]
		self.cam_0127 = [0, 1, 2, 7]

		self.selected_keys = [k for k in keys_to_use]
		assert len(self.selected_keys) > 0

		self.file_paths = [os.path.join(self.path, k) for k in self.selected_keys]

		for num_i in range(0, len(self.selected_keys)):
			file_path = os.path.join(self.path, self.selected_keys[num_i])
			self.hdfs[num_i] = h5py.File(file_path, 'r', swmr=True)
			print('read file: ', os.path.join(self.path, self.selected_keys[num_i]))
			assert self.hdfs[num_i].swmr_mode

		self.idx_to_kv = []
		for num_i in range(0, len(self.selected_keys)):
			this_sub = self.selected_keys[num_i].split('.')[0]
			n = self.hdfs[num_i][image_key].shape[0]

			if full_light_only:
				self.light_meta = OmegaConf.load(osp.join(osp.dirname(osp.realpath(__file__)), '../data/light_metadata.yaml'))
				n_full_light = self.light_meta[this_sub]['full_light_frame_end_idx'] * 18
				n_all = self.light_meta[this_sub]['num_total_frames'] * 18
				print(" only load full-light frames, {} / {} frames. while n = {}".format(n_full_light, n_all, n))
				n = n_full_light

			if self.camera_tag == 'all':
				self.idx_to_kv += [(num_i, i) for i in range(n)]
			elif self.camera_tag == 'cam00':
				self.idx_to_kv += [(num_i, i) for i in range(n) if (i % 18) == 0]
			elif self.camera_tag == 'small_cam_8':
				self.idx_to_kv += [(num_i, i) for i in range(n) if (i % 18) in self.small_cam_8]
			elif self.camera_tag == 'small_cam_8_c':
				self.idx_to_kv += [(num_i, i) for i in range(n) if (i % 18) in self.small_cam_8_c]
			elif self.camera_tag == 'small_cam_2':
				self.idx_to_kv += [(num_i, i) for i in range(n) if (i % 18) in self.small_cam_2]
			elif self.camera_tag == 'small_cam_2_c':
				self.idx_to_kv += [(num_i, i) for i in range(n) if (i % 18) in self.small_cam_2_c]
			elif self.camera_tag == 'cam_0127':
				self.idx_to_kv += [(num_i, i) for i in range(n) if (i % 18) in self.cam_0127]

		for num_i in range(0, len(self.hdfs)):            
			if self.hdfs[num_i]:
				self.hdfs[num_i].close()
				self.hdfs[num_i] = None

		self.transform = transforms.Compose([
			transforms.ToPILImage(),
			transforms.ToTensor(),
			transforms.Normalize(mean=transform_Normalize['mean'], std=transform_Normalize['std']),
		])

		self.__hdfs = None
		self.hdf = None

	def __len__(self):
		return len(self.idx_to_kv)

	def __del__(self):
		for num_i in range(0, len(self.hdfs)):
			if self.hdfs[num_i]:
				self.hdfs[num_i].close()
				self.hdfs[num_i] = None

	@property
	def archives(self):
		if self.__hdfs is None:  # lazy loading here!
			self.__hdfs = [h5py.File(h5_path, "r", swmr=True) for h5_path in self.file_paths]
		return self.__hdfs
	
	# def calculate_eye_height(self, image):
	# 	image = image[..., ::-1]  # Convert BGR to RGB
	# 	# img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	# 	results = face_mesh.process(image)
	# 	if not results.multi_face_landmarks:
	# 		return 7.5  # If no face is detected, return None
	# 	height, width, _ = image.shape
	# 	face_landmarks = results.multi_face_landmarks[0]
	# 	left_eye_upper = face_landmarks.landmark[159]  # Upper eyelid of left eye
	# 	left_eye_lower = face_landmarks.landmark[145]  # Lower eyelid of left eye
	# 	right_eye_upper = face_landmarks.landmark[386]  # Upper eyelid of right eye
	# 	right_eye_lower = face_landmarks.landmark[374]  # Lower eyelid of right eye
	# 	left_eye_upper_coords = (int(left_eye_upper.x * width), int(left_eye_upper.y * height))
	# 	left_eye_lower_coords = (int(left_eye_lower.x * width), int(left_eye_lower.y * height))
	# 	right_eye_upper_coords = (int(right_eye_upper.x * width), int(right_eye_upper.y * height))
	# 	right_eye_lower_coords = (int(right_eye_lower.x * width), int(right_eye_lower.y * height))
	# 	left_eye_height = math.sqrt(
	# 	(left_eye_upper_coords[0] - left_eye_lower_coords[0]) ** 2 +
	# 	(left_eye_upper_coords[1] - left_eye_lower_coords[1]) ** 2
	# 	)
	# 	right_eye_height = math.sqrt(
	# 	(right_eye_upper_coords[0] - right_eye_lower_coords[0]) ** 2 +
	# 	(right_eye_upper_coords[1] - right_eye_lower_coords[1]) ** 2
	# 	)
	# 	mean_eye_height = (left_eye_height + right_eye_height) / 2
	# 	return mean_eye_height

	def preprocess_image(self, image):
		image = image.astype(np.float32)
		if self.color_type == 'bgr':
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_AREA)
		image = self.transform(image.astype(np.uint8))
		return image

	def __getitem__(self, index):
		key, idx = self.idx_to_kv[index]
		self.hdf = self.archives[key]
		assert self.hdf.swmr_mode

		image = self.hdf[self.image_key][idx, :]
		gaze_label = self.hdf['face_gaze'][idx].astype('float') if 'face_gaze' in self.hdf else np.array([0, 0]).astype('float')
		head_label = self.hdf['face_head_pose'][idx].astype('float') if 'face_head_pose' in self.hdf else np.array([0, 0]).astype('float')
		eye_height_label = self.hdf['eye_height'][idx].astype('float')

		entry = {
			'image': self.preprocess_image(image),
			'gaze': gaze_label,
			'head': head_label,
			'eye_height': eye_height_label,  # Add eye height to the entry
			'key': key,
			'index': index
		}

		if "augmented_face_patch" in self.hdf:
			entry['augmented_image'] = self.preprocess_image(self.hdf['augmented_face_patch'][idx, :][0])

		return entry
