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
        self.f_oo_vis = nn.Sequential(
                                        nn.Linear(1024*2, 1024),
                                        nn.Linear(1024,512),
                                        nn.ReLU()
                                    ).to(self.config['device'])

    def prepare_f_oo_vis_input(
                                self, num_rels, object_branch_output, 
                                context_branch_output, num_obj, 
                                obj_pairs
                              ):

        object_branch_output_dim = object_branch_output.shape[-1]
        context_branch_output_dim = context_branch_output.shape[-1]

        tot_num_rels = int(torch.sum(num_rels))

        input_f_oo_vis = torch.zeros((
                                    tot_num_rels, 
                                    object_branch_output_dim + context_branch_output_dim
                                    ), device = self.config['device'])
        
        batch_size = num_rels.shape[0]

                
        # verify this again carefully
        for curr_batch in range(batch_size):
            
            curr_num_rels = int(num_rels[curr_batch])
            object_index_offset = int( torch.sum(num_obj[:curr_batch]) )
            relation_index_offset = int( torch.sum(num_rels[:curr_batch]) )
            
            for j in range(curr_num_rels):

                obj_ind_0, obj_ind_1 = obj_pairs[curr_batch, j]

                obj_vec_0 = object_branch_output[ int(obj_ind_0.item()) + object_index_offset ]
                obj_vec_1 = object_branch_output[ int(obj_ind_1.item()) + object_index_offset ]

                temp_obj_vector = aggregate(obj_vec_0, obj_vec_1, 'mean')
                temp_context_vector = context_branch_output[curr_batch]
                
                temp_relation_index = int(relation_index_offset + j)
                
                input_f_oo_vis[temp_relation_index, :] = torch.cat((
                                                                    temp_obj_vector, 
                                                                    temp_context_vector), 0)
        
        return input_f_oo_vis
    
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
        
        object_branch_output = self.object_branch(
                                                    frame_feature_map, 
                                                    bboxes, 
                                                    num_obj, 
                                                    image_dimension
                                                 )

        context_branch_output = self.context_branch( frame_feature_map )
        
        # collect all the relevant object pairs into one large tensor for further processing

        
        input_f_oo_vis = self.prepare_f_oo_vis_input(
                                                    num_rels, object_branch_output, 
                                                    context_branch_output, num_obj, 
                                                    obj_pairs
                                                    )

        f_oo_vis_output = self.f_oo_vis(input_f_oo_vis)
        
        return object_branch_output, f_oo_vis_output