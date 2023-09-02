import torch
import torch.nn as nn
import torchvision

class ObjectBranch_vsgnet(torch.nn.Module):

    def __init__(self, config):

        super(ObjectBranch_vsgnet, self).__init__()
        self.config = config

        self.Conv_objects = nn.Sequential(
				nn.Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
				nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
				nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
				nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
				nn.Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False),
				nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
				nn.ReLU(inplace=False)
				                        ).to(self.config['device'])

        self.pool_size = tuple(self.config['roi_pool_size'])
        self.spatial_scale = self.config['spatial_scale']
        self.sampling_ratio = self.config['sampling_ratio']

        self.residual_identity_projection = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.ReLU()
        )
        
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


    def forward(self, frame_feature_map, bboxes, num_obj, image_dimension):
        
        # Get roi_values of the objects
        num_frames = int(bboxes.shape[-2])
        central_frame_index = int(num_frames/2)
        
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

        ##Objects##
        residual_objects = roi_pool_objects
        res_objects = self.Conv_objects(roi_pool_objects) + self.residual_identity_projection(residual_objects)
        res_av_objects = self.obj_pool(res_objects)
        
        # flattening the output
        out2_objects = res_av_objects.view(res_av_objects.size()[0], -1)
        
        return out2_objects