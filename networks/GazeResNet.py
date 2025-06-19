import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from .modules.resnet import resnet18, resnet50




class GazeRes18(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature =  resnet18(pretrained=True)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, 2)

    def forward(self, x_in):
        x = self.feature(x_in)
        # x = self.avgpool(x) # the avgpool is already included in self.feature
        x = x.view(x.size(0), -1)

        gaze = self.fc(x)
        return {'pred_gaze': gaze}


class GazeRes50(nn.Module):
    def __init__(self, estimate_head=False):
        super().__init__()
        self.estimate_head = estimate_head
    
        self.feature =  resnet50(pretrained=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        if self.estimate_head:
            self.fc = nn.Linear(2048, 4)
        else:
            self.fc = nn.Linear(2048, 2)
        
    def forward(self, x_in,  trained=True):
        x = self.feature(x_in)
        # x = self.avgpool(features)  # the avgpool is already included in self.feature
        x = x.view(x.size(0), -1)
        if self.estimate_head:
            output = self.fc(x)  # (batchsize, 4)
            pred_head = output[:,:2]
            pred_gaze = output[:,2:]
            return {'pred_head': pred_head, 'pred_gaze': pred_gaze}
        else:
            gaze = self.fc(x)
            return {'pred_gaze': gaze}