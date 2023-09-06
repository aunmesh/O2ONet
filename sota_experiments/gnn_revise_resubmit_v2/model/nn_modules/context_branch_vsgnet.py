import torch
import torch.nn as nn
import torch.nn.functional as F

class ContextBranch_vsgnet(torch.nn.Module):

    def __init__(self, config):
        
        super(ContextBranch_vsgnet, self).__init__()
        self.config = config

        self.Conv_context=nn.Sequential(
				nn.Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
				nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
				nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
				nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
				nn.Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False),
				nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
				nn.ReLU(inplace=False),
				)
        
        self.residual_identity_projection = nn.Sequential(
				nn.Conv2d(2048, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False),
				nn.ReLU(inplace=False),
				)
	
    def forward(self, frame_feature_map):
        
        x,y = frame_feature_map.size()[2],frame_feature_map.size()[3]	
        residual_context = frame_feature_map
        res_context = self.Conv_context(residual_context) + self.residual_identity_projection(residual_context)
        res_av_context = F.avg_pool2d(res_context, kernel_size=(x,y), stride=(1,1), padding=0)
        out_context = res_av_context.view(res_av_context.size()[0], -1)
        
        return out_context