import torch
import torch.nn as nn
import torchvision
from utils.utils import aggregate

class ObjectBranch_drg(torch.nn.Module):

    def __init__(self, config):

        super(ObjectBranch_drg, self).__init__()
        self.config = config

        self.lin_object_layers = []
        dimensions = self.config['lin_obj_dim']
        
        for i in range( len(dimensions) - 1 ):

            curr_d = dimensions[i]
            next_d = dimensions[i+1]

            temp_fc_layer = nn.Linear(curr_d, next_d)
            self.lin_object_layers.append(temp_fc_layer)
            self.lin_object_layers.append(nn.ReLU())

        self.lin_obj_net = nn.Sequential(*self.lin_object_layers).to(self.config['device'])

        self.pool_size = tuple(self.config['roi_pool_size'])
        self.spatial_scale = self.config['spatial_scale']
        self.sampling_ratio = self.config['sampling_ratio']

        self.pooler = torchvision.ops.RoIAlign(
                                                self.pool_size, 
                                                self.spatial_scale, 
                                                self.sampling_ratio
                                               ).to(self.config['device'])    

        # Global average pooling
        self.obj_pool = nn.AvgPool2d(self.pool_size, padding=0, stride=(1,1)).to(self.config['device'])


    def scale_bboxes(self, bboxes, frame_feature_map_shape, image_dimension):
        
        # Get the output from subbranch for all the objects
        im_height, im_width = image_dimension
        fmap_height, fmap_width = frame_feature_map_shape[-2:]
        
        height_scale = fmap_height/(im_height*1.0)
        width_scale = fmap_width/(im_width*1.0)
        bboxes_scaled = bboxes.clone()

        bboxes_scaled[:,:,0] = bboxes[:,:,0] * width_scale
        bboxes_scaled[:,:,2] = bboxes[:,:,2] * width_scale
        bboxes_scaled[:,:,1] = bboxes[:,:,1] * height_scale
        bboxes_scaled[:,:,3] = bboxes[:,:,3] * height_scale
        
        return bboxes_scaled
    
    def pair_features(
                    self, num_rels, object_branch_output, 
                    num_obj, obj_pairs
                    ):

        object_branch_output_dim = object_branch_output.shape[-1]
        tot_num_rels = int(torch.sum(num_rels))

        paired_features = torch.zeros((
                                    tot_num_rels, 
                                    object_branch_output_dim
                                    ),device = self.config['device'])

        batch_size = num_rels.shape[0]

        for curr_batch in range(batch_size):
            
            curr_num_rels = int(num_rels[curr_batch])
            object_index_offset = int( torch.sum(num_obj[:curr_batch]) )
            relation_index_offset = int( torch.sum(num_rels[:curr_batch]) )
            
            for j in range(curr_num_rels):

                obj_ind_0, obj_ind_1 = obj_pairs[curr_batch, j]

                obj_vec_0 = object_branch_output[ int(obj_ind_0.item()) + object_index_offset ]
                obj_vec_1 = object_branch_output[ int(obj_ind_1.item()) + object_index_offset ]

                temp_obj_vector = aggregate(obj_vec_0, obj_vec_1, 'mean')
                
                temp_relation_index = int(relation_index_offset + j)
                
                paired_features[temp_relation_index, :] = temp_obj_vector
                
        return paired_features


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
        
        # getting the bboxes for the central frame
        bboxes_scaled = self.scale_bboxes(bboxes, frame_feature_map.shape, image_dimension)

        # Get 10*10 roi pool of all the regions 
        bboxes_list = torch.split(bboxes_scaled, 1, dim=0)
        bboxes_list = [b.squeeze(0) for b in bboxes_list]
        
        # removing unnecessary objects is important to reduce computation downstream
        for i, b in enumerate(bboxes_list):
            temp_num_obj = int(num_obj[i])
            bboxes_list[i] = b[:temp_num_obj]
        
        roi_pool_objects = self.pooler(frame_feature_map, bboxes_list)

        # swapping dimensions for linear layer
        roi_pool_objects = roi_pool_objects.permute(0, 2, 3, 1)
        ##Objects##
        res_objects = self.lin_obj_net(roi_pool_objects)
        res_objects = res_objects.permute(0, 3, 1, 2)
        res_objects_pooled = self.obj_pool(res_objects)
        
        # flattening the output
        res_objects_flattened = res_objects_pooled.view(res_objects_pooled.size()[0], -1)
        
        # pairing the individual object features using an aggregation operation
        res_objects_paired = self.pair_features( num_rels, 
                                                 res_objects_flattened,
                                                 num_obj, obj_pairs
                                                )

        return res_objects_paired