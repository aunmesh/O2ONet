import torch
import torch.nn as nn

from utils.utils import aggregate

class SpatialBranch_vsgnet(torch.nn.Module):
    
    def __init__(self, config):
        
        super(SpatialBranch_vsgnet, self).__init__()
        
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
	
    def scale_bbox(self, box, image_dimension, req_dim):

        scale_factor = [ req_dim[i]/image_dimension[i] for i in range(len(image_dimension)) ]
        
        scaled_box = box.clone()
        
        scaled_box[0]*=scale_factor[0]
        scaled_box[2]*=scale_factor[0]

        scaled_box[1]*=scale_factor[1]
        scaled_box[3]*=scale_factor[1]
        
        return scaled_box
        
        
    def prepare_input_spatial_conv(self, bboxes, obj_pairs, num_rels, image_dimension):
        '''
        generates the bounding box region
        '''
        tot_num_rels = int(torch.sum(num_rels))
        input_sp_map = torch.zeros(tot_num_rels, 2, 64, 64, device=self.config['device'])
        batch_size = bboxes.shape[0]
        
        curr_relation_index = 0

        for curr_batch in range(batch_size):
            
            curr_num_rels = int(num_rels[curr_batch])

            for j in range(curr_num_rels):

                obj_ind_0, obj_ind_1 = obj_pairs[curr_batch, j]
                obj_ind_0, obj_ind_1 = int(obj_ind_0), int(obj_ind_1)

                bbox_0 = bboxes[curr_batch, obj_ind_0]
                bbox_1 = bboxes[curr_batch, obj_ind_1]
                
                bbox_0_scaled = self.scale_bbox(bbox_0, image_dimension, [64, 64])
                bbox_1_scaled = self.scale_bbox(bbox_1, image_dimension, [64, 64])                
                
                temp_x0, temp_x1 = int(bbox_0_scaled[1]), int(bbox_0_scaled[3])
                temp_y0, temp_y1 = int(bbox_0_scaled[0]), int(bbox_0_scaled[2])
                 
                input_sp_map[curr_relation_index, 0, temp_x0:temp_x1, temp_y0:temp_y1] = 1
                
                temp_x0, temp_x1 = int(bbox_1_scaled[1]), int(bbox_1_scaled[3])
                temp_y0, temp_y1 = int(bbox_1_scaled[0]), int(bbox_1_scaled[2])
                 
                input_sp_map[curr_relation_index, 1, temp_x0:temp_x1, temp_y0:temp_y1] = 1
                
                curr_relation_index += 1
        
        return input_sp_map


    def forward(self, data_item):
        
        bboxes = data_item['bboxes']
        obj_pairs = data_item['object_pairs']
        num_rels = data_item['num_relation']
        
        # Get the output from subbranch for all the objects
        frame_width = data_item['metadata']['frame_width'][0]
        frame_height = data_item['metadata']['frame_height'][0]
        image_dimension = [frame_width, frame_height]

        
        # prepare input to spatial conv
        input_sp_map = self.prepare_input_spatial_conv(
                                                        bboxes, obj_pairs, 
                                                        num_rels, image_dimension
                                                       )

        # pass it through the conv
        spatial_feature_map = self.conv_sp_map(input_sp_map)
        spatial_feature_map = spatial_feature_map.view(spatial_feature_map.size()[0], -1)
        spatial_feature_map = self.spmap_up(spatial_feature_map)

        return spatial_feature_map