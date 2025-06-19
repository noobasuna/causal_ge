import os
import numpy as np
import h5py
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
from typing import List
from omegaconf import OmegaConf
import os.path as osp
import dlib

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Initialize Dlib's face detector and facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/home/tpei0009/STSTNet/shape_predictor_68_face_landmarks.dat")  # Adjust path to your setup

def create_left_eye_mask(image, landmarks):
    mask = np.zeros_like(image)
    points = []
    for i in range(36, 42):
        points.append((landmarks.part(i).x, landmarks.part(i).y))
    points = np.array(points, dtype=np.int32)
    cv2.fillConvexPoly(mask, points, (255, 255, 255))
    return mask

def create_right_eye_mask(image, landmarks):
    mask = np.zeros_like(image)
    points = []
    for i in range(42, 48):
        points.append((landmarks.part(i).x, landmarks.part(i).y))
    points = np.array(points, dtype=np.int32)
    cv2.fillConvexPoly(mask, points, (255, 255, 255))
    return mask

def mask_both_eyes(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) == 0:
        return img  # If no face is detected, return the original image

    for face in faces:
        landmarks = predictor(gray, face)
        left_eye_mask = create_left_eye_mask(img, landmarks)
        right_eye_mask = create_right_eye_mask(img, landmarks)
        combined_mask = cv2.bitwise_or(left_eye_mask, right_eye_mask)
        eye_region = cv2.bitwise_and(img, combined_mask)

        return eye_region

class XGazeDataset(Dataset):
    def __init__(self, 
                 dataset_path: str, 
                 color_type: str,
                 keys_to_use: List[str] = None, 
                 data_name=None, 
                 image_size: int = 224,
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

        assert camera_tag in ['all', 'cam00', 'small_cam_2', 'small_cam_8', 'small_cam_2_c', 'small', 'cam_0127']  
        self.camera_tag = camera_tag
        self.small_cam_8 = [0, 1, 2, 3, 6, 7, 8, 9]
        self.small_cam_2 = [0, 1]
        self.cam_0127 = [0, 1, 2, 7]

        self.selected_keys = [k for k in keys_to_use]
        assert len(self.selected_keys) > 0
        
        self.file_paths = [os.path.join(self.path, k) for k in self.selected_keys]

        for num_i in range(len(self.selected_keys)):
            file_path = os.path.join(self.path, self.selected_keys[num_i])
            self.hdfs[num_i] = h5py.File(file_path, 'r', swmr=True)
            print('Read file:', os.path.join(self.path, self.selected_keys[num_i]))
            assert self.hdfs[num_i].swmr_mode

        self.idx_to_kv = []
        for num_i in range(len(self.selected_keys)):
            this_sub = self.selected_keys[num_i].split('.')[0]
            n = self.hdfs[num_i][image_key].shape[0]
            
            if full_light_only:
                self.light_meta = OmegaConf.load(osp.join(osp.dirname(osp.realpath(__file__)), '../data/light_metadata.yaml'))
                n_full_light = self.light_meta[this_sub]['full_light_frame_end_idx'] * 18
                print("Only load full-light frames, {} frames.".format(n_full_light))
                n = n_full_light

            if self.camera_tag == 'all':
                self.idx_to_kv += [(num_i, i) for i in range(n)]
            elif self.camera_tag == 'cam00':
                self.idx_to_kv += [(num_i, i) for i in range(n) if (i % 18) == 0]
            elif self.camera_tag == 'small_cam_8':
                self.idx_to_kv += [(num_i, i) for i in range(n) if (i % 18) in self.small_cam_8]
            elif self.camera_tag == 'small_cam_2':
                self.idx_to_kv += [(num_i, i) for i in range(n) if (i % 18) in self.small_cam_2]
            elif self.camera_tag == 'cam_0127':
                self.idx_to_kv += [(num_i, i) for i in range(n) if (i % 18) in self.cam_0127]

        for num_i in range(len(self.hdfs)):            
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
        for num_i in range(len(self.hdfs)):
            if self.hdfs[num_i]:
                self.hdfs[num_i].close()
                self.hdfs[num_i] = None

    @property
    def archives(self):
        if self.__hdfs is None:  # Lazy loading here!
            self.__hdfs = [h5py.File(h5_path, "r", swmr=True) for h5_path in self.file_paths]
        return self.__hdfs

    def preprocess_image(self, image):
        image = mask_both_eyes(image) 
        cv2.imwrite('/home/tpei0009/gaze-demo/gaze-demo/example_masked_wo_eye.jpg', image) #check

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

        entry = {
            'image': self.preprocess_image(image),
            'gaze': gaze_label,
            'head': head_label,
            'key': key,
            'index': index
        }
        
        if "augmented_face_patch" in self.hdf:
            entry['augmented_image'] = self.preprocess_image(self.hdf['augmented_face_patch'][idx, :][0])
        
        return entry
