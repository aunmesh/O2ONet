import torch
import torch.nn as nn

from model.nn_modules.object_branch_ican import ObjectBranch_ican
from model.nn_modules.context_branch_ican import ContextBranch_ican
from utils.utils import aggregate


class VisualBranch_ican(torch.nn.Module):
    
    def __init__(self, config):
        
        super(VisualBranch_ican, self).__init__()
        
        self.config = config

        self.object_branch = ObjectBranch_ican(config)
        self.context_branch = ContextBranch_ican(config)
        
        self.object_lin_transform = nn.Sequential(
                                                nn.Linear(2048, 512),
                                                nn.ReLU()
                                                 ).to(self.config['device'])
        
        self.context_lin_transform = nn.Sequential(
                                                nn.Linear(512, 1024),
                                                nn.ReLU()
                                                 ).to(self.config['device'])

        self.concat_lin_transform = nn.Sequential(
                                                nn.Linear(3072, 1024),
                                                nn.ReLU()
                                                 ).to(self.config['device'])


    def pair_features(
                    self, num_rels, object_features, 
                    num_obj, obj_pairs
                    ):

        object_features_dim = object_features.shape[-1]
        tot_num_rels = int(torch.sum(num_rels))

        paired_features = torch.zeros((
                                    tot_num_rels, 
                                    object_features_dim
                                    ),device = self.config['device'])

        batch_size = num_rels.shape[0]

        for curr_batch in range(batch_size):
            
            curr_num_rels = int(num_rels[curr_batch])
            object_index_offset = int( torch.sum(num_obj[:curr_batch]) )
            relation_index_offset = int( torch.sum(num_rels[:curr_batch]) )
            
            for j in range(curr_num_rels):

                obj_ind_0, obj_ind_1 = obj_pairs[curr_batch, j]

                obj_vec_0 = object_features[ int(obj_ind_0.item()) + object_index_offset ]
                obj_vec_1 = object_features[ int(obj_ind_1.item()) + object_index_offset ]

                temp_obj_vector = aggregate(obj_vec_0, obj_vec_1, 'mean')
                
                temp_relation_index = int(relation_index_offset + j)
                
                paired_features[temp_relation_index, :] = temp_obj_vector
                
        return paired_features

    def iCAN(self, object_query, context_key, context_val):
        context_key_flattened = context_key.view(context_key.size()[0], -1)
        object_query_ = object_query.unsqueeze(1)
        temp_mul = object_query_ * context_key_flattened
        dot = torch.sum(temp_mul, dim=0)

        attention = torch.nn.functional.softmax(dot)
        
        context_val_flattened = context_val.view(context_val.size()[0], -1)
        context_val_modulated = context_val_flattened * attention
        
        attended_context = torch.sum(context_val_modulated, dim=1)
        
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
        
        obj_branch_output, obj_slicing = self.object_branch(data_item)
                                                   
        obj_querys = self.object_lin_transform(obj_branch_output)
        
        context_branch_output_key, context_branch_output_val = self.context_branch(
                                                                    frame_feature_map
                                                                                  )
        
        num_obj_querys = obj_querys.shape[0]  # must be same as torch.sum(num_obj)
        
        obj_attended_context = torch.zeros((num_obj_querys, 512), device=self.config['device'])
        
        for n in range(num_obj_querys):
            
            batch_idx = int(obj_slicing[n])
            
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
        obj_concatenated_transformed = self.concat_lin_transform(obj_concatenated)
        
        obj_concatenated_paired = self.pair_features(num_rels, 
                                                     obj_concatenated_transformed,
                                                     num_obj, obj_pairs)
        
        return obj_concatenated_paired