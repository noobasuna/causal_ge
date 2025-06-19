import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.cit import chisq, fisherz, kci, d_separation, mv_fisherz,gsq

from .modules.resnet import resnet18, resnet50

class Causal_GazeRes18(nn.Module):
    def __init__(self):
        super().__init__()
        # Create three separate feature extractors
        self.feature1 = resnet18(pretrained=True)
        self.feature2 = resnet18(pretrained=True)
        self.feature3 = resnet18(pretrained=True)
        
        # Fully connected layers for each input
        self.fc1 = nn.Linear(512, 2)  # For first input
        self.fc2 = nn.Linear(512, 2)  # For second input
        self.fc3 = nn.Linear(512, 2)  # For third input
        self.gaze_predictions = []  # Initialize a list to store gaze predictions

    def forward(self, x1, x2, x3):
        # Separate feature extraction for each input
        f1 = self.feature1(x1)
        f2 = self.feature2(x2)
        f3 = self.feature3(x3)
        
        # Flatten features
        f1 = f1.view(f1.size(0), -1)
        f2 = f2.view(f2.size(0), -1)
        f3 = f3.view(f3.size(0), -1)
        
        # Separate predictions for each input
        pred_gaze_1 = self.fc1(f1)
        pred_gaze_2 = self.fc2(f2)
        pred_gaze_3 = self.fc3(f3)
        
        # Return separate output dictionaries
        return (
            {'pred_gaze': pred_gaze_1},  # prediction for first input
            {'pred_gaze': pred_gaze_2},  # prediction for second input
            {'pred_gaze': pred_gaze_3}   # prediction for third input
        )
    
    def save_gaze_predictions(self, file_path='gaze_predictions.npy'):
        # Concatenate all gaze predictions and save to a .npy file
        all_gaze_predictions = np.concatenate(self.gaze_predictions, axis=0)
        np.save(file_path, all_gaze_predictions)
        print(f'Gaze predictions saved to {file_path}')

class GazeRes50(nn.Module):
    def __init__(self, estimate_head=False):
        super().__init__()
        self.estimate_head = estimate_head
    
        # Create three separate feature extractors
        self.feature1 = resnet50(pretrained=True)
        self.feature2 = resnet50(pretrained=True)
        self.feature3 = resnet50(pretrained=True)
        
        # Adjust fully connected layer to handle combined features
        if self.estimate_head:
            self.fc = nn.Linear(2048 * 3, 4)
        else:
            self.fc = nn.Linear(2048 * 3, 2)
        
    def forward(self, x1, x2, x3, trained=True):
        # Extract features from each input
        f1 = self.feature1(x1)
        f2 = self.feature2(x2)
        f3 = self.feature3(x3)
        
        # Flatten features
        f1 = f1.view(f1.size(0), -1)
        f2 = f2.view(f2.size(0), -1)
        f3 = f3.view(f3.size(0), -1)
        
        # Concatenate features
        combined_features = torch.cat([f1, f2, f3], dim=1)
        
        # Classification based on head estimation flag
        if self.estimate_head:
            output = self.fc(combined_features)  # (batchsize, 4)
            pred_head = output[:,:2]
            pred_gaze = output[:,2:]
            return {'pred_head': pred_head, 'pred_gaze': pred_gaze}
        else:
            gaze = self.fc(combined_features)
            return {'pred_gaze': gaze}