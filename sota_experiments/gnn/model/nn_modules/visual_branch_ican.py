import torch
import torch.nn as nn

from model.nn_modules.object_branch_vsgnet import ObjectBranch_vsgnet
from model.nn_modules.context_branch_vsgnet import ContextBranch_vsgnet
from utils.utils import aggregate


class VisualBranch_vsgnet(torch.nn.Module):
    
    def __init__(self, config):
        
        super(VisualBranch_vsgnet, self).__init__()
        
        self.config = config

        self.object_branch = ObjectBranch_vsgnet(config)
        self.context_branch = ContextBranch_vsgnet(config)
        
        self.object_lin_transform = nn.Sequential(
                                                nn.Linear(2048, 512),
                                                nn.ReLU()
                                                 ).to(self.config['device'])
        
        self.context_lin_transform = nn.Sequential(
                                                nn.Linear(512, 1024),
                                                nn.ReLU()
                                                 ).to(self.config['device'])


    def iCAN(self, object_query, context_key, context_val):
        
        context_key_flattened = context_key.view(context_key.size()[0], -1)
        
        dot = torch.sum(context_key * object_query, dim=1)

        attention = torch.nn.functional.softmax(dot)
        
        context_val_flattened = context_val.flatten()
        context_val_modulated = context_val_flattened * attention
        
        attended_context = torch.sum(context_val_modulated, dim=0)
        
        return attended_context

    
    def forward(self, data_item):
        
        frame_feature_map = data_item['frame_deep_features']
        bboxes = data_item['bboxes']
        num_obj = data_item['num_obj']
        obj_pairs = data_item['object_pairs']
        num_rels = data_item['num_relation']
        
        # Get the output from subbranch for all the objects
        frame_width = data_item['metadata']['frame_width'][0]
        frame_height = data_item['metadata']['frame_height'][0]
        image_dimension = [frame_width, frame_height]
        
        obj_branch_output, obj_slicing = self.object_branch(
                                                    frame_feature_map, 
                                                    bboxes, 
                                                    num_obj, 
                                                    image_dimension
                                                 )
        obj_querys = self.object_lin_transform(obj_branch_output)
        
        context_branch_output_key, context_branch_output_val = self.context_branch(
                                                                    frame_feature_map
                                                                                  )
        
        num_obj_querys = obj_querys.shape[0]  # must be same as torch.sum(num_obj)
        
        obj_attended_context = torch.zeros((num_obj_querys, 512), device=self.config['device'])
        
        for n in range(num_obj_querys):
            
            batch_idx = obj_slicing[n]
            
            # MUST BE as C, H, W
            req_feature_map_key = context_branch_output_key[batch_idx]
            req_feature_map_val = context_branch_output_val[batch_idx]            
            
            temp_obj_query = obj_querys[n]
            temp_obj_context = self.iCAN(temp_obj_query, 
                                         req_feature_map_key, 
                                         req_feature_map_val )
            
            obj_attended_context[n, :] = temp_obj_context
        
        obj_attended_context_transformed = self.context_lin_transform(obj_attended_context)

        obj_concatenated = torch.cat((obj_branch_output, obj_attended_context_transformed), dim=1)        
        
        return obj_concatenated