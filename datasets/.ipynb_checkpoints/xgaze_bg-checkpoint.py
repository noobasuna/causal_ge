import os, random
import os.path as osp
from glob import glob
import numpy as np
import h5py
import cv2
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
from typing import List
from omegaconf import OmegaConf

# Default mean and standard deviation for normalization
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class XGazeDataset(Dataset):
    def __init__(self, 
                dataset_path: str, 
                color_type,
                keys_to_use: List[str] = None, 
                data_name=None, 
                image_size:int=224,
                image_key='face_patch',
                gaze_key='face_gaze',
                transform_Normalize={'mean': MEAN, 'std': STD}, 
                camera_tag='all',
                full_light_only=False):
        
        self.path = dataset_path
        self.hdfs = {}
        self.image_key = image_key
        self.gaze_key = gaze_key
        self.image_size = (image_size, image_size)
        
        assert color_type in ['rgb', 'bgr']
        self.color_type = color_type
        self.camera_tag = camera_tag

        # Camera definitions for filtering
        self.small_cam_8 = [0,1,2,3, 6,7,8,9]
        self.small_cam_2 = [0,1]
        self.cam_0127 = [0,1,2,7]
        
        # Selected keys to use from dataset
        self.selected_keys = [k for k in keys_to_use]
        assert len(self.selected_keys) > 0
        self.file_paths = [os.path.join(self.path, k) for k in self.selected_keys]

        # Load the HDF5 files
        for num_i in range(0, len(self.selected_keys)):
            file_path = os.path.join(self.path, self.selected_keys[num_i])
            self.hdfs[num_i] = h5py.File(file_path, 'r', swmr=True)
            print('Read file: ', file_path)
            assert self.hdfs[num_i].swmr_mode

        # Mapping from full-data index to key and person-specific index
        self.idx_to_kv = []
        for num_i in range(0, len(self.selected_keys)):
            n = self.hdfs[num_i][image_key].shape[0]
            if self.camera_tag == 'all':
                self.idx_to_kv += [(num_i, i) for i in range(n)]
            elif self.camera_tag == 'cam00':
                self.idx_to_kv +=  [(num_i, i) for i in range(n) if (i % 18) == 0 ]
            elif self.camera_tag == 'small_cam_8':
                self.idx_to_kv +=  [(num_i, i) for i in range(n) if (i % 18) in self.small_cam_8 ]
            elif self.camera_tag == 'small_cam_2':
                self.idx_to_kv +=  [(num_i, i) for i in range(n) if (i % 18) in self.small_cam_2 ]
            elif self.camera_tag == 'cam_0127':
                self.idx_to_kv +=  [(num_i, i) for i in range(n) if (i % 18) in self.cam_0127 ]

        for num_i in range(0, len(self.hdfs)):            
            if self.hdfs[num_i]:
                self.hdfs[num_i].close()
                self.hdfs[num_i] = None

        # Transform pipeline
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),  # Convert pixel value from [0, 255] to [0, 1]
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
        if self.__hdfs is None:  # Lazy loading
            self.__hdfs = [h5py.File(h5_path, "r", swmr=True) for h5_path in self.file_paths]
        return self.__hdfs

    def detect_face(self, image):
        # Load the Haar Cascade classifier for face detection
        face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
        
        # Convert image to grayscale for face detection
        gray_image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # If a face is detected, return the bounding box of the first face
        if len(faces) > 0:
            return faces[0]  # Return the first detected face (x, y, width, height)
        else:
            return None  # No face detected

    def preprocess_image(self, image, face_bbox=None):
        image = image.astype(np.float32)

        # Convert to RGB if the image is in BGR format
        if self.color_type == 'bgr':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize the image
        image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_AREA)

        # If no bounding box is provided, detect the face using Haar Cascade
        if face_bbox is None:
            face_bbox = self.detect_face(image)  # Call the face detection function

        # If a face is detected, mask the face region
        if face_bbox is not None:
            x, y, w, h = face_bbox  # Assuming face_bbox is a tuple (x, y, width, height)
            image[y:y+h, x:x+w] = 0  # Set face region pixels to zero
        
        # Convert to PIL image and apply transformations
        image = self.transform(image.astype(np.uint8))
        return image

    def __getitem__(self, index):
        key, idx = self.idx_to_kv[index]
        self.hdf = self.archives[key]
        assert self.hdf.swmr_mode

        # Extract image, gaze, and head data
        image = self.hdf[self.image_key][idx, :]
        gaze_label = self.hdf['face_gaze'][idx].astype('float') if 'face_gaze' in self.hdf else np.array([0, 0]).astype('float')
        head_label = self.hdf['face_head_pose'][idx].astype('float') if 'face_head_pose' in self.hdf else np.array([0, 0]).astype('float')

        # If bounding box is not provided, detect the face
        if 'face_bbox' in self.hdf:
            face_bbox = self.hdf['face_bbox'][idx]  # If bounding box exists in dataset
        else:
            face_bbox = None  # Set to None if bounding box is not available

        # Preprocess the image, with or without face masking
        image = self.preprocess_image(image, face_bbox)

        entry = {
            'image': image,
            'gaze': gaze_label,
            'head': head_label,
            'key': key,
            'index': index
        }

        if "augmented_face_patch" in self.hdf:
            entry['augmented_image'] = self.preprocess_image(self.hdf['augmented_face_patch'][idx, :][0])

        return entry