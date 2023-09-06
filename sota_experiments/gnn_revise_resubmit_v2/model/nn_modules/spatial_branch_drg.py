import torch
import torch.nn as nn

from utils.drg_utils import *

class SpatialBranch_drg(torch.nn.Module):
    
    def __init__(self, config):
        
        super(SpatialBranch_drg, self).__init__()
        
        self.config = config
        
        self.conv_sp_map=nn.Sequential(
                        nn.Conv2d(2, 64, kernel_size=(5, 5)),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=(2, 2)),
                        nn.Conv2d(64, 32, kernel_size=(5,5)),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=(2, 2)),
                        # nn.AvgPool2d((13,13),padding=0,stride=(1,1))
                                    )

        self.spmap_up=nn.Sequential(
                    nn.Linear(5408,4800),
                    nn.ReLU(),
                    )
	

    def prepare_input_spatial_conv(self, bboxes, num_obj):
        '''
        generates the bounding box region
        '''
        
        batch_size = bboxes.shape[0]
        tot_num_combinations = 0
        all_combinations = []
        
        for b in range(batch_size):
            curr_num_obj = int(num_obj[b])
            curr_num_combinations = int(( curr_num_obj * (curr_num_obj-1) )/2.0)
            tot_num_combinations+=curr_num_combinations
            all_combinations.append(curr_num_combinations)
        
       
        slicing_tensor = torch.zeros((tot_num_combinations, 3), device=self.config['device'])
        input_sp_map = torch.zeros((tot_num_combinations, 2, 64, 64), device=self.config['device'])
        curr_index = 0

        for curr_batch in range(batch_size):
            
            curr_num_obj = int(num_obj[curr_batch])
            curr_num_combinations = int((curr_num_obj * (curr_num_obj-1) )/2.0)

            for i in range(curr_num_obj):
                for j in range(i+1, curr_num_obj):

                    bbox_0 = bboxes[curr_batch, i]
                    bbox_1 = bboxes[curr_batch, j]
                    
                    input_sp_map[ curr_index, :, :, :] = get_sp(
                                                                bbox_0, bbox_1, 
                                                                self.config['device']
                                                                )
                    
                    slicing_tensor[curr_index, 0] = curr_batch
                    slicing_tensor[curr_index, 1] = i
                    slicing_tensor[curr_index, 2] = j
                    
                    curr_index += 1
        
        return input_sp_map, slicing_tensor

    def forward(self, data_item):
        
        bboxes = data_item['bboxes']
        num_obj = data_item['num_obj']        
        
        # prepare input to spatial conv
        input_sp_map, slicing_tensor = self.prepare_input_spatial_conv( bboxes, num_obj)

        # pass it through the conv
        spatial_feature_map = self.conv_sp_map(input_sp_map)
        spatial_feature_map = spatial_feature_map.view(spatial_feature_map.size()[0], -1)
        spatial_feature_map = self.spmap_up(spatial_feature_map)

        return spatial_feature_map, slicing_tensor