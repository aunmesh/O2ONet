import torch
import torch.nn as nn
import torchvision
from utils.utils import aggregate

class ObjectBranch_ican(torch.nn.Module):

    def __init__(self, config):

        super(ObjectBranch_ican, self).__init__()
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

        self.pool_size = (7,7)
        self.spatial_scale = self.config['spatial_scale']
        self.sampling_ratio = self.config['sampling_ratio']

        self.pooler = torchvision.ops.RoIAlign(
                                                self.pool_size, 
                                                self.spatial_scale, 
                                                self.sampling_ratio
                                               ).to(self.config['device'])    

        # Global average pooling
        self.obj_pool = nn.AvgPool2d(self.pool_size, padding=0, stride=(1,1)).to(self.config['device'])

        self.residual_transform = nn.Sequential(
                                                nn.Linear(2048, 2048),
                                                nn.ReLU()
                                                )

        self.res5 = nn.Sequential(
				nn.Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
				nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
				nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding='same',bias=False),
				nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
				nn.Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False),
				nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
				nn.ReLU(inplace=False)
				                        ).to(self.config['device'])

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

        # Get 7*7 roi pool of all the regions 
        bboxes_list = torch.split(bboxes_scaled, 1, dim=0)
        bboxes_list = [b.squeeze(0) for b in bboxes_list]

        tot_num_obj = int(torch.sum(num_obj))
        slicing_tensor = torch.zeros((tot_num_obj), device=self.config['device'])
        lower_index = 0
        upper_index = 0

        # removing unnecessary objects is important to reduce computation downstream
        for i, b in enumerate(bboxes_list):

            temp_num_obj = int(num_obj[i])
            bboxes_list[i] = b[:temp_num_obj]

            upper_index += temp_num_obj
            slicing_tensor[lower_index:upper_index] = i
            lower_index = upper_index
        
        roi_pool_objects = self.pooler(frame_feature_map, bboxes_list)

        # swapping dimensions for linear layer
        roi_pool_objects = roi_pool_objects.permute(0, 2, 3, 1)
        roi_pool_residual = self.residual_transform(roi_pool_objects).permute(0,3,1,2)
        
        # reswapping for res5 conv layers
        roi_pool_objects = roi_pool_objects.permute(0,3,1,2)
        temp_res5 = self.res5(roi_pool_objects)
        res5_objects =  temp_res5 + roi_pool_residual
        
        res5_objects_gap = self.obj_pool(res5_objects)

        # ##Objects##
        # res_objects = self.lin_obj_net(roi_pool_objects)
        # res_objects = res_objects.permute(0, 3, 1, 2)
        # res_objects_pooled = self.obj_pool(res_objects)
        
        # flattening the output
        res5_objects_flattened = res5_objects_gap.view(res5_objects_gap.size()[0], -1)
        

        return res5_objects_flattened, slicing_tensor