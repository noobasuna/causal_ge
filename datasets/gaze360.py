
from torch.utils.data import Dataset
from torchvision import transforms, utils
import h5py, cv2, random
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image
from typing import List

from glob import glob

def keep_background(image_in):
	image = image_in.copy()
	h, w, c = image.shape
	empty_img = np.zeros((h, w), dtype=np.uint8)
	RED, GREEN, BLUE = (2, 1, 0)
	reds = image[:, :, RED]
	greens = image[:, :, GREEN]
	blues = image[:, :, BLUE]
	
	mask = (greens > (reds + 20)) & (greens > (blues + 20))
	empty_img[mask] = 200
	
	kernel = np.ones((3, 3), np.uint8)
	for _ in range(5):
		empty_img = cv2.erode(empty_img, kernel, iterations=3)
		empty_img = cv2.dilate(empty_img, kernel, iterations=3)
	
	mask = (empty_img != 200)
	image[mask] = (0, 0, 0)  # Set the detected foreground to black

	return image

def wrap_transforms(image_transforms_type, image_size):

	if image_transforms_type == 'basic_generation':
		MEAN = [0.5,0.5,0.5]
		STD = [0.5,0.5,0.5]
		return transforms.Compose([
				transforms.ToPILImage(),
				transforms.ToTensor(),
				transforms.Normalize(mean=MEAN, std=STD)
			])
	elif image_transforms_type == 'basic_imagenet':
		MEAN = [0.485, 0.456, 0.406]
		STD = [0.229, 0.224, 0.225]
		return transforms.Compose([
				transforms.ToPILImage(),
				transforms.ToTensor(),
				transforms.Normalize(mean=MEAN, std=STD)
			])
	else:
		raise NotImplementedError

"""
  data_name: gaze360_224_train <-- NOTE: data_name does not matter
  dataset_path: null <------------ NOTE: set this to your path <...>/gaze360_normalized_h5/train
  color_type: bgr
  transform_type: 'basic_imagenet'
  image_size: 224
  keys_to_use: [] ## <-- set this to empty list will automatically read all the h5 files in the dataset_path
"""


"""
  data_name: gaze360_224_test <--- NOTE: data_name does not matter
  dataset_path: null <------------ NOTE: set this to your path <...>/gaze360_normalized_h5/test
  color_type: bgr
  transform_type: 'basic_imagenet'
  image_size: 224
  keys_to_use: [] ## <-- set this to empty list will automatically read all the h5 files in the dataset_path
"""

class Gaze360Dataset(Dataset):
	def __init__(self, 
				dataset_path: str, 
				color_type,
				keys_to_use: List[str] = None, 
				data_name=None, 
				image_size:int=224,
				transform_type='basic_imagenet',
				image_key='face_patch',
				gaze_key='face_gaze',
				# race_label='race',
				# eye_height_label='eye_height',
				# average_eye_height_label = 'average_eye_height',
				transform_Normalize=None
				):
		super().__init__()
		self.dataset_path = dataset_path
		self.hdfs = {}
		self.data_name = data_name
		self.image_key = image_key
		self.gaze_key = gaze_key
		self.image_size = (image_size, image_size)
		# self.race_label = race_label
		# self.eye_height_label= eye_height_label
		# self.average_eye_height_label = average_eye_height_label

		assert color_type in ['rgb', 'bgr']
		self.color_type = color_type
		self.transform = wrap_transforms(transform_type, image_size=image_size)


		#### -------------------------------------------------------- read the h5 files ------------------------------------------------------- 
		if keys_to_use is None or len(keys_to_use) == 0:
			keys_to_use = [os.path.basename(f) for f in sorted(glob(os.path.join(self.dataset_path, '*.h5')))]

		self.selected_keys = [k for k in keys_to_use]
		assert len(self.selected_keys) > 0
		self.file_paths = [os.path.join(self.dataset_path, k) for k in self.selected_keys]
		for num_i in range(0, len(self.selected_keys)):
			file_path = os.path.join(self.dataset_path, self.selected_keys[num_i]) # the subdirectories: train, test are not used in MPIIFaceGaze and MPII_Rotate
			self.hdfs[num_i] = h5py.File(file_path, 'r', swmr=True)
			# print('read file: ', os.path.join(self.dataset_path, self.selected_keys[num_i]))
			assert self.hdfs[num_i].swmr_mode
		####----------------------------------------------------------------------------------------------------------------------------------- 

		self.build_idx_to_kv()
		for num_i in range(0, len(self.hdfs)):            
			if self.hdfs[num_i]:
				self.hdfs[num_i].close()
				self.hdfs[num_i] = None

		self.__hdfs = None
		self.hdf = None

	def build_idx_to_kv(self):
		self.idx_to_kv = []
		self.key_idx_dict = {}

		for num_i in range(0, len(self.selected_keys)):
			p_key = self.selected_keys[num_i].split('.')[0]  ##p00
			n = self.hdfs[num_i][self.image_key].shape[0] 
			indices = np.arange(0, n)
			self.idx_to_kv += [(num_i, i) for i in indices]
			self.key_idx_dict[p_key] = [i for i in indices]


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
	
	def preprocess_bg_image(self, image):
		image = keep_background(image)  # Apply keep_background here
		# cv2.imwrite('/home/tpei0009/gaze-demo/gaze-demo/datasets/xgaze_bg.jpg', image)
		image = image.astype(np.float32)
		# if self.color_type == 'bgr':
		# 	image = image[..., ::-1]
		image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_AREA)
		image = self.transform(image.astype(np.uint8))
		return image	
	
	def preprocess_image(self, image):
		image = image.astype(np.float32)
		if self.color_type == 'bgr':
			# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			image = image[..., ::-1]
		# image = image[..., ::-1]
		if image.shape[0] != self.image_size[0] or image.shape[1] != self.image_size[1]:
			image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_AREA)
		cv2.imwrite('/home/tpei0009/gaze-demo/gaze-demo/gaze360_ex.jpg', image)
		image = self.transform(image.astype(np.uint8))
		return image
	
	def __getitem__(self, index):
		key, idx = self.idx_to_kv[index]
		self.hdf = self.archives[key]
		image = self.hdf[self.image_key][idx]
		head_label = self.hdf['face_head_pose'][idx].astype('float') if 'face_head_pose' in self.hdf else np.array([0,0]).astype('float')
		gaze_label = self.hdf[self.gaze_key][idx].astype('float') if self.gaze_key in self.hdf else np.array([0,0]).astype('float')
		# eye_height_label = self.hdf['eye_height'][idx].astype('float')
		# Handle average_eye_height with shape (1,)
		# average_eye_height_label = self.hdf['average_eye_height'][0].astype('float')  # Access the single value
		# average_eye_height_label = np.full((1,), average_eye_height_label)  # Broadcast to match batch size if needed

		# race_str = self.hdf[self.race_label][()].decode('utf-8')		
		# print(race_str)
		# race_mapping = {
		# 	'white': 0,
		# 	'black': 1,
		# 	'asian': 2,
		# 	'indian': 3,
		# 	'middle eastern': 4,
		# 	'latino hispanic': 5,
		# }
		# race_label = race_mapping.get(race_str.lower(), -1)  # Default to -1 if race not found


		entry = {
			'image': self.preprocess_image(image),
			# 'bg_image': self.preprocess_bg_image(image),
			# 'eye_height': eye_height_label,  
			# 'average_eye_height': average_eye_height_label,
			'gaze': gaze_label,
			'head': head_label,
			'key': idx,
			'index':index,
			# 'race': np.array(race_label).astype('float')
		}
		return entry
	