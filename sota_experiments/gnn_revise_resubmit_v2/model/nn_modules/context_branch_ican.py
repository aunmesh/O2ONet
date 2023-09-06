import torch
import torch.nn as nn
import torch.nn.functional as F

class ContextBranch_ican(torch.nn.Module):

    def __init__(self, config):
        
        super(ContextBranch_ican, self).__init__()
        self.config = config

        self.Conv_context_key=nn.Sequential(
				nn.Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
				nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
				nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
				nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
				nn.ReLU(inplace=False),
				)

        self.Conv_context_value=nn.Sequential(
				nn.Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
				nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
				nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
				nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
				nn.ReLU(inplace=False),
				)


	
    def forward(self, frame_feature_map):
        
        res_context_key = self.Conv_context_key(frame_feature_map)
        res_context_value = self.Conv_context_value(frame_feature_map)
        
        return res_context_key, res_context_value