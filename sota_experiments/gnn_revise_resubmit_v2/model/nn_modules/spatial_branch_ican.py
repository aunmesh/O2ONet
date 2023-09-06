import torch
import torch.nn as nn

from utils.drg_utils import *

class SpatialBranch_ican(torch.nn.Module):
    
    def __init__(self, config):
        
        super(SpatialBranch_ican, self).__init__()
        
        self.config = config
        
        self.conv_sp_map=nn.Sequential(
                        nn.Conv2d(2, 64, kernel_size=(5, 5)),
                        nn.MaxPool2d(kernel_size=(2, 2)),
                        nn.Conv2d(64, 32, kernel_size=(5,5)),
                        nn.MaxPool2d(kernel_size=(2, 2)),
                        nn.AvgPool2d((13,13),padding=0,stride=(1,1))
                                    )

        self.spmap_up=nn.Sequential(
                                    nn.Linear(32,512),
                                    nn.ReLU(),
                                    )

        
    def prepare_input_spatial_conv(self, bboxes, num_rels, obj_pairs):
        '''
        generates the bounding box region
        '''
        
        batch_size = bboxes.shape[0]
        tot_num_rels = int(torch.sum(num_rels))
        input_sp_map = torch.zeros((tot_num_rels, 2, 64, 64), device=self.config['device'])

        curr_index = 0

        for curr_batch in range(batch_size):
            
            temp_num_rel = int( num_rels[curr_batch] )

            for i in range(temp_num_rel):
                
                obj_ind_0, obj_ind_1 = obj_pairs[curr_batch, i, :]
                
                obj_ind_0 = int(obj_ind_0)
                obj_ind_1 = int(obj_ind_1)
                
                bbox_0 = bboxes[curr_batch, obj_ind_0]
                bbox_1 = bboxes[curr_batch, obj_ind_1]
                
                input_sp_map[ curr_index, :, :, :] = get_sp(
                                                            bbox_0, bbox_1, 
                                                            self.config['device']
                                                            )
                
                curr_index += 1
        
        return input_sp_map

    def forward(self, data_item):
        
        bboxes = data_item['bboxes']
        num_obj = data_item['num_obj']        
        num_rels = data_item['num_relation']
        obj_pairs = data_item['object_pairs']
        # prepare input to spatial conv
        input_sp_map = self.prepare_input_spatial_conv( bboxes, num_rels, obj_pairs)

        # pass it through the conv
        spatial_feature_map = self.conv_sp_map(input_sp_map)
        spatial_feature_map = spatial_feature_map.view(spatial_feature_map.size()[0], -1)
        spatial_feature_map = self.spmap_up(spatial_feature_map)

        return spatial_feature_map