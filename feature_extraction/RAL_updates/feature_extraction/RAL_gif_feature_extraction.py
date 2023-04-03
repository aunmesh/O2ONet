anno_path = '/workspace/work/misc/O2ONet/data/annotations_minus_unavailable_yt_vids.pkl'

import pickle as pkl
import numpy as np

f = open(anno_path, 'rb')
anno = pkl.load(f)
f.close()


# Tracker with recovery
def tracker(frames, main_bbox_tb):
    import cv2
    import sys
    
    image_height, image_width,_ = frames[0].shape

    main_bbox_wh = (
                    main_bbox_tb[0], 
                    main_bbox_tb[1],
                    main_bbox_tb[2]-main_bbox_tb[0],
                    main_bbox_tb[3]-main_bbox_tb[1]
                    )
    (major_ver, minor_ver, subminor_ver) = cv2.__version__.split('.')


    # Set up tracker.
    # Instead of MIL, you can also use

    tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    tracker_type = tracker_types[-1]

    if int(minor_ver) < 3:
        tracker = cv2.Tracker_create(tracker_type)
    else:
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
            tracker_rev = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
            tracker_rev = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
            tracker_rev = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv2.legacy_TrackerTLD.create()
            tracker_rev = cv2.legacy_TrackerTLD.create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.legacy_TrackerMedianFlow.create()
            tracker_rev = cv2.legacy_TrackerMedianFlow.create()
        if tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()
            tracker_rev = cv2.TrackerGOTURN_create()
        if tracker_type == 'MOSSE':
            tracker = cv2.legacy_TrackerMOSSE.create()
            tracker_rev = cv2.legacy_TrackerMOSSE.create()
        if tracker_type == "CSRT":
            tracker = cv2.TrackerCSRT_create()
            tracker_rev = cv2.TrackerCSRT_create()

    num_frames = len(frames)

    central_index = int((num_frames - 1)/2)
    window_size = int(num_frames/2)

    central_frame = frames[central_index]

    # Initialize tracker with first frame and bounding box
    
    ok = tracker.init(central_frame, main_bbox_wh)
    bboxes_forward = []

    for i in range(window_size):

        # Read a new frame
        frame = frames[central_index + 1 + i]        

        # Update tracker
        ok, bbox_wh = tracker.update(frame)
        if not ok:
            print(bbox_wh)
        # add to the bbox list
        if ok:
            bbox_tb = [ bbox_wh[0], bbox_wh[1], bbox_wh[0] + bbox_wh[2], bbox_wh[1] + bbox_wh[3] ]
            import numpy as np

            bbox_tb[0], bbox_tb[2] = np.clip(bbox_tb[0],0, image_width-1), np.clip(bbox_tb[2],0, image_width-1)
            bbox_tb[1], bbox_tb[3] = np.clip(bbox_tb[1],0, image_height-1), np.clip(bbox_tb[3],0, image_height-1)

            bboxes_forward.append(bbox_tb)
            # # Tracking success
            # p1 = (int(bbox[0]), int(bbox[1]))
            # p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            # cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
        else :
            print("Tracking Failure")
            if len(bboxes_forward) > 0:
                bboxes_forward.append(bboxes_forward[-1])
            else:
                bboxes_forward.append(main_bbox_tb)
            # bbox_wh = bboxes
            # return 0
            # Tracking failure
            # cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

    # Initialize tracker with first frame and bounding box
    ok = tracker_rev.init(central_frame, main_bbox_wh)
    bboxes_backward = []
    for i in range(window_size):
        
        # Read a new frame
        frame = frames[central_index - 1 - i]        
        
        # Update tracker
        ok, bbox_wh = tracker_rev.update(frame)

        # Add to the bbox list
        if ok:
            import numpy as np
            bbox_tb = [ bbox_wh[0], bbox_wh[1], bbox_wh[0] + bbox_wh[2], bbox_wh[1] + bbox_wh[3] ]

            bbox_tb[0], bbox_tb[2] = np.clip(bbox_tb[0],0, image_width-1), np.clip(bbox_tb[2],0, image_width-1)
            bbox_tb[1], bbox_tb[3] = np.clip(bbox_tb[1],0, image_height-1), np.clip(bbox_tb[3],0, image_height-1)

            bboxes_backward.append(bbox_tb)
            # # Tracking success
            # p1 = (int(bbox[0]), int(bbox[1]))
            # p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            # cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
        else:
            print("Tracking Failure")
            if len(bboxes_backward) > 0:
                bboxes_backward.append(bboxes_backward[-1])
            else:
                bboxes_backward.append(main_bbox_tb)
            # return 0
            # Tracking failure
            # cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

    del tracker
    del tracker_rev
    bboxes_backward_reversed = bboxes_backward[-1::-1]
    all_bbox = bboxes_backward_reversed + [main_bbox_tb] + bboxes_forward
    
    return all_bbox

# For Geometric Feature ( based on bbox dimensions )
def geometric_feature(bbox, im_width, im_height):
    '''
    In Modeling Context Between Objects for Referring Expression Understanding, ECCV 2016
    [x_min/W, y_min/H, x_max/W, y_max/H, bbox_area/image_area]
    
    The annotation are given in Image Coordinate system (X is horizontal & Y is vertical ,(0,0) top left)
    The features are calculated in Image Coordinate System as well
    '''
    x_min = bbox[0]   
    y_min = bbox[1]

    x_max = bbox[2]
    y_max = bbox[3]

    bbox_width = x_max - x_min
    bbox_height = y_max - y_min
    
    area_bbox = bbox_width * bbox_height
    area_image = im_width * im_height
    
    feature = [x_min/im_width, y_min/im_height, x_max/im_width, y_max/im_height, area_bbox/area_image]
    import numpy as np
    import torch
    feature = np.asarray(feature, dtype=np.float32)
    feature = torch.from_numpy(feature)
    return feature


# For 2d cnn based deep bbox features
import torch.nn as nn
import torchvision.models as models_torchvision

import pynvml

def get_memory_free_MiB(gpu_index):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu_index))
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return mem_info.free // 1024 ** 2


class ImageFeatureExtractor(nn.Module):
    
    """
    Object feature extractor
    """
    
    def __init__(self, submodule, layer, device, deep_net):
    
        """
        input the object detector module and the layer
        number on which we want to extract features
        """
        
        super(ImageFeatureExtractor, self).__init__()
        
        self.pretrain_model = submodule
        self.layer = layer
        
        model = models_torchvision.resnet152(pretrained=True)
        self.feature_extract_net = nn.Sequential(*list(model.children())[0:8])
        self.feature_extract_net = self.feature_extract_net.eval()
        self.pretrain_model = None

        from torchvision import transforms
        self.transform_module = transforms.Compose([
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                                        std=[0.229, 0.224, 0.225]),
                                                    ])


    def forward(self, images, device):
        
        with torch.no_grad():
            image_tensors = [self.transform_module(image).unsqueeze(0) for image in images]            
            image_concatenated_tensor = torch.cat(image_tensors, dim=0).to(device)
            
            feature = self.feature_extract_net(image_concatenated_tensor)
            feature_cpu = feature.detach().cpu()
            
            del feature
            del image_concatenated_tensor   # preventing memory leak
            del image_tensors
            
            self.feature_extract_net.zero_grad()

        return feature_cpu

from time import time
import numpy as np
import torch

from torchvision.transforms import ToTensor
def extract_image_deep_feature_faster(images, feature_extractor, device):

    image_feature = feature_extractor(images, device)
    return image_feature

import torchvision

# Doesn't need correction
def roi_align(feature_map, boxes):
    
    pooler = torchvision.ops.RoIAlign(output_size=(7, 7), spatial_scale = 1.0, sampling_ratio=1)
    boxes_list = [boxes]
    output = pooler(feature_map.unsqueeze(0), boxes_list)

    return output

import torch.nn.functional as F 
import torch

# Corrected
def extract_bbox_deep_features_faster(bboxes, im_shape, fmap, device):
    '''
    bboxes: tensor with bbox coords
    '''
    
    im_width_annotation, im_height_annotation = im_shape

    _, fmap_height, fmap_width, __ = fmap.shape
    fmap_scale_width, fmap_scale_height = (fmap_width*1.0)/im_width_annotation, (fmap_height*1.0)/im_height_annotation

    fmap_device = fmap.device

    from copy import deepcopy as copy
    boxes = copy(bboxes)
    
    boxes[... ,0] *= fmap_scale_width
    boxes[... ,2] *= fmap_scale_width
    boxes[... ,1] *= fmap_scale_height
    boxes[... ,3] *= fmap_scale_height
    
    boxes = boxes.to(fmap_device)
    
    num_frames = fmap.shape[0]
    all_frame_bbox_features = []
    for n in range(num_frames):
        bbox_features = roi_align(fmap[n], boxes[n])       
        bbox_features = F.avg_pool2d(bbox_features, (7,7)).squeeze(2).squeeze(2)
        all_frame_bbox_features.append(bbox_features.unsqueeze(0))
    
    all_frame_bbox_features = torch.cat(all_frame_bbox_features, dim=0)
        
    return all_frame_bbox_features



# For mIoU and distance
from shapely.geometry import Polygon

# Corrected
def calculate_iou(box_1, box_2):

    '''
    boxes in [min_x, min_y, max_x, max_y] format
    '''
    # if torch.sum(box_1 == box_2) == 4:
    # return 1

    b1_min_x, b1_min_y = box_1[0], box_1[1]
    b1_max_x, b1_max_y = box_1[2], box_2[3]

    b2_min_x, b2_min_y = box_2[0], box_2[1]
    b2_max_x, b2_max_y = box_2[2], box_2[3]


    b1 = [[b1_min_x, b1_min_y], [b1_min_x, b1_max_y], [b1_max_x, b1_max_y], [b1_max_x, b1_min_y]]
    b2 = [[b2_min_x, b2_min_y], [b2_min_x, b2_max_y], [b2_max_x, b2_max_y], [b2_max_x, b2_min_y]]

    poly_1 = Polygon(b1)
    poly_2 = Polygon(b2)

    i_area = poly_1.intersection(poly_2).area
    u_area = poly_1.union(poly_2).area
    
    iou = i_area / u_area
    
    return iou

# Corrected
def calculate_distance_normalized(box_1, box_2, im_width, im_height):
    
    '''
    boxes in [min_x, min_y, max_x, max_y] format
    '''

    b1_c_x = (box_1[0] + box_1[2]) * 0.5
    b1_c_y = (box_1[1] + box_1[3]) * 0.5

    b2_c_x = (box_2[0] + box_2[2]) * 0.5
    b2_c_y = (box_2[1] + box_2[3]) * 0.5

    b1_x, b1_y = b1_c_x/im_width, b1_c_y/im_height
    b2_x, b2_y = b2_c_x/im_width, b2_c_y/im_height
    
    # normalized distance in 0 to 1
    dis = np.sqrt( (b1_x-b2_x)**2 + (b1_y-b2_y)**2 ) / np.sqrt(2)

    return dis


# For relative spatial features
import numpy as np
import torch
from shapely.geometry import Polygon

# Corrected
def box_deltas(subject_box, object_box):
    '''
    boxes in [centre_x, centre_y, width, height] format
    '''

    s_width = subject_box[2] - subject_box[0]
    s_height = subject_box[3] - subject_box[1]
    
    o_width = object_box[2] - object_box[0]
    o_height = object_box[3] - object_box[1]

    s_centre_x = subject_box[0] + (s_width/2)
    s_centre_y = subject_box[1] + (s_height/2)

    o_centre_x = object_box[0] + (o_width/2)
    o_centre_y = object_box[1] + (o_height/2)
    
    t_so_x = (s_centre_x - o_centre_x)/s_width
    t_so_y = (s_centre_y - o_centre_y)/s_height
    
    t_so_w = torch.log(s_width/o_width)
    t_so_h = torch.log(s_height/o_height)
    
    t_os_x = (o_centre_x - s_centre_x)/o_width
    t_os_y = (o_centre_y - s_centre_y)/o_height
    
    data = [t_so_x, t_so_y, t_so_w, t_so_h, t_os_x, t_os_y]

    return torch.FloatTensor(data)


def get_union_box(box_1, box_2):

    '''
    boxes in [min_x, min_y, max_x, max_y] format
    '''

    b1_min_x, b1_min_y = box_1[0], box_1[1]
    b1_max_x, b1_max_y = box_1[2], box_2[3]

    b2_min_x, b2_min_y = box_2[0], box_2[1]
    b2_max_x, b2_max_y = box_2[2], box_2[3]

    bu_min_x, bu_min_y = min(b1_min_x, b2_min_x), min(b1_min_y, b2_min_y)
    bu_max_x, bu_max_y = max(b1_max_x, b2_max_x), max(b1_max_y, b2_max_y)
  
    return [bu_min_x, bu_min_y, bu_max_x, bu_max_y]

def calculate_distance(box_1, box_2):
    
    '''
    boxes in [min_x, min_y, max_x, max_y] format
    '''

    b1_c_x = (box_1[0] + box_1[2]) * 0.5
    b1_c_y = (box_1[1] + box_1[3]) * 0.5

    b2_c_x = (box_2[0] + box_2[2]) * 0.5
    b2_c_y = (box_2[1] + box_2[3]) * 0.5

    dis = np.sqrt( (b1_c_x-b2_c_x)**2 + (b1_c_y-b2_c_y)**2 )

    return dis


def relative_spatial_features(bbox_1, bbox_2, im_width, im_height):
    
    bbox_1[0]/=im_width
    bbox_1[2]/=im_width
    bbox_1[1]/=im_height
    bbox_1[3]/=im_height

    bbox_2[0]/=im_width
    bbox_2[2]/=im_width
    bbox_2[1]/=im_height
    bbox_2[3]/=im_height
    
    relative_features = torch.zeros(20, dtype=torch.float32)
    
    subject_box = bbox_1
    object_box = bbox_2

    union_box = get_union_box(subject_box, object_box)

    relative_features[:6] = box_deltas(subject_box=subject_box, object_box=object_box)
    relative_features[6:12] = box_deltas(subject_box=subject_box, object_box=union_box)
    relative_features[12:18] = box_deltas(subject_box=object_box, object_box=union_box)
    relative_features[18] = calculate_iou(subject_box, object_box)
    relative_features[19] = calculate_distance(subject_box, object_box)
    
    return relative_features



# For i3d based bbox features

import numpy as np
import torch
import torchvision
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection import FasterRCNN
from torch import nn
import torchvision.transforms as transforms
import pytorchvideo.models as models
import torch.nn.functional as F
from collections import OrderedDict
from torch import nn

class FeatureExtractor(nn.Module):

    def __init__(self, submodule, layer):
        super(FeatureExtractor, self).__init__()
        self.pretrain_model = submodule
        self.layer = layer
        
        self.layer_list = list(self.pretrain_model._modules['blocks']._modules.keys())
        print(list(self.pretrain_model._modules['blocks']._modules))
        output_layer = self.layer_list[self.layer]  # just change the number of the layer to get the output

        self.children_list = []
        for (name, comp_layer) in self.pretrain_model._modules['blocks'].named_children():
            self.children_list.append(comp_layer)
            if name == output_layer:
                break
        #print(self.children_list)
        self.feature_extrac_net = nn.Sequential(*self.children_list)
        self.pretrain_model = None

    def forward(self, image):
        feature = self.feature_extrac_net(image)
        return feature

from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import  NormalizeVideo

from pytorchvideo.transforms import (
    ApplyTransformToKey,
    UniformTemporalSubsample
)


import numpy as np

def read_gif(gif_path):
    """read gif and return dictionary as key ['video'] and value the tensor of size(CxTxHXW)"""

    video = EncodedVideo.from_path(gif_path)
    video = video.get_clip(0, 5) # get_clip fetches the clip from starting time to ending time

    return video

# Doesn't need correction
def roi_align_i3d(feature_map, boxes):
    
    pooler = torchvision.ops.RoIAlign(output_size=(1, 1), spatial_scale = 1.0, sampling_ratio=1)
    boxes_list = [boxes]
    output = pooler(feature_map, boxes_list)

    return output

def roi_align_custom(feature_map, boxes, im_width, im_height):
    '''
    feature_map : [B,C,T,H,W] B - Batch size (expected 1)
    boxes: [N, T, 4] N is number of objects
    '''
    
    fmap_height, fmap_width = feature_map.shape[3:]
    boxes[0]/=im_width
    boxes[2]/=im_width

    boxes[1]/=im_height
    boxes[3]/=im_height

    boxes[0]*=fmap_width
    boxes[2]*=fmap_width

    boxes[1]/=fmap_height
    boxes[3]/=fmap_height
    # output['bboxes'] = torch.zeros(max_num_obj,len(frames),4, dtype=torch.float)
    # uniform temporal subsample selects 1,2,3,4,5,6,7,8,9,11
    boxes = boxes[:,0:9:2,:]
    time_steps = boxes.shape[1]
    
    roi_align_res = []
    for t in range(time_steps):
        temp_fmap = feature_map[:,:,t,:,:]
        temp_boxes = boxes[:,t,:]
        temp_res = roi_align_i3d(temp_fmap, temp_boxes)
        roi_align_res.append(temp_res)
    
    return roi_align_res


# For motion feature calculation

def calculate_motion_feature(geom_feat_1, geom_feat_2):
    return geom_feat_1 - geom_feat_2


# Master Feature Generator
import torch
def master_feature_generator(annotation, gif_folder, cnn_feature_extractor, 
                             i3d_feature_extractor, i3d_transform, device):

    # Getting details to load the GIF
    yt_id = annotation['metadata']['yt_id']
    frame_index = annotation['metadata']['frame no.']

    temp = int(int(gif_folder.split('_')[-1])/2)
    window_size = temp

    # Loading the gif    
    # getting the file location
    filename = yt_id + '_' + str(frame_index) + '_' + str(window_size) + '.gif'
    import os
    file_location = os.path.join(gif_folder, filename)
    import cv2

    # getting the frames
    vid = cv2.VideoCapture(file_location)
    frames = []
    frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(frame_count):
        success, frame = vid.read()
        frames.append(frame)
    central_frame = frames[window_size]

    # Sanity Check    
    assert window_size == (len(frames) - 1)/2, "Possible issue, please check"

    # Output Dictionary
    output = {}
    output['legend'] = {}
    
    # Adding the metadata
    output['metadata'] = annotation['metadata']
    im_height, im_width, _ = frames[0].shape    # NumPy has num rows, num cols which is height and width according to opencv conventions
    output['metadata']['frame_width'] = im_width
    output['metadata']['frame_height'] = im_height
    
    # total num of annotated objects 
    output['num_obj'] = len(list(annotation['bboxes'].keys()))

    # saving bbox co-ordinates of objects according to their key in bboxes field
    max_num_obj = 12

    # bounding box coordinates. not normalized. for image width and height see the metadata
    output['bboxes'] = torch.zeros(max_num_obj,len(frames),4, dtype=torch.float)
    bbox_keys = annotation['bboxes'].keys()
    
    for key in bbox_keys:
        key_val = int(key)
        temp_bbox = annotation['bboxes'][key]['bbox']
        
        tracked_bboxes = tracker(frames, temp_bbox)
        tracked_bboxes = torch.from_numpy(np.asarray(tracked_bboxes, dtype=float))
        output['bboxes'][key_val,:,:] = tracked_bboxes

    from copy import deepcopy as copy
    
    # saving relations in tensors

    # maps to transform text to indices
    cr_map = {'Contact': 0, 'No Contact': 1, 'None of these': 2, '': 2}
    lr_map = {'Below/Above': 0, 'Behind/Front': 1, 'Left/Right': 2, 'Inside': 3, 'None of these': 4, '': 4}
    mr_map = {'Holding': 0, 'Carrying': 1, 'Adjusting': 2, 'Rubbing': 3, 'Sliding': 4, 'Rotating': 5, 'Twisting': 6,
              'Raising': 7, 'Lowering': 8, 'Penetrating': 9, 'Moving Toward': 10, 'Moving Away': 11, 
              'Negligible Relative Motion': 12, 'None of these': 13, '': 13}

    max_num_rels = 15

    # tensor storing relations between objects at the corresponding index in object_pairs key
    output['lr'] = torch.zeros(max_num_rels, 5)
    output['mr'] = torch.zeros(max_num_rels, 14)
    output['cr'] = torch.zeros(max_num_rels, 3)
    
    # object indices between which the corresponding relation is annotated
    output['object_pairs'] = torch.zeros(max_num_rels, 2)
    
    # reading relations and saving them to the tensors
    for i, rel in enumerate(annotation['relations']):

        object_pairs = rel[0]

        mr = rel[1]['mr']
        lr = rel[1]['lr']
        cr = rel[1]['scr']

        for r in mr:
            temp_val = mr_map[r]
            output['mr'][i, temp_val] = 1
        for r in lr:
            temp_val = lr_map[r]
            output['lr'][i, temp_val] = 1
        for r in cr:
            temp_val = cr_map[r]
            output['cr'][i, temp_val] = 1

        output['object_pairs'][i] = torch.from_numpy(np.asarray(object_pairs,dtype=float))

    # total number of relations and hence the total number of object pairs as well
    output['num_relation'] = len(annotation['relations'])

    # Now we have bounding boxes, metadata, relations, number of objects, number of relations    
    # image features - cnn features for bboxes, bbox coordinate based features, relative feature, miou, distance,

    # bbox coordinate based features
    output['geometric_feature'] = torch.zeros(max_num_obj, len(frames), 5, dtype=float)
    
    for f in range(len(frames)):
        for i in range( int(output['num_obj']) ):
            temp_bbox = copy(output['bboxes'][i, f])
            output['geometric_feature'][i, f] = geometric_feature(temp_bbox, im_width, im_height)

    import pynvml
    def get_memory_free_MiB(gpu_index):
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu_index))
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return mem_info.free // 1024 ** 2
    
    # 2d cnn feature map
    temp_fmap = extract_image_deep_feature_faster( frames, cnn_feature_extractor, device)
    
    temp_cpu = temp_fmap.detach().cpu()
    output['2d_cnn_feature_map'] = temp_cpu
    
    # # 2d cnn based features for the bounding boxes of central frame

    # # window_size is also the index of the central frame
    all_frame_bboxes = copy(output['bboxes'])[:, :output['num_obj']]
    temp_bbox_feat = extract_bbox_deep_features_faster( all_frame_bboxes, 
                                                        [im_width, im_height], 
                                                        temp_fmap, device
                                                    )

    output['object_2d_cnn_feature'] = torch.zeros(len(frames), max_num_obj, 2048, dtype=float)

    output['object_2d_cnn_feature'][:, :output['num_obj'], :] = temp_bbox_feat

    del temp_bbox_feat
    del temp_fmap

    # miou and distance of bounding boxes
    output['iou'] = torch.zeros(max_num_obj, max_num_obj, len(frames))

    for f in range(len(frames)):
        for i in range( int(output['num_obj']) ):
                for j in range( int(output['num_obj']) ):
                    
                    temp_box_1 = copy(output['bboxes'])[i, f]
                    temp_box_2 = copy(output['bboxes'])[j, f]
                    output['iou'][i, j, f] = calculate_iou(temp_box_1, temp_box_2)
                        

    output['distance'] = torch.zeros(max_num_obj, max_num_obj, len(frames))

    for f in range(len(frames)):
        for i in range( int(output['num_obj']) ):
                for j in range( int(output['num_obj']) ):

                    temp_box_1 = copy(output['bboxes'])[i, f]
                    temp_box_2 = copy(output['bboxes'])[j, f]
                    output['distance'][i, j, f] = calculate_distance_normalized(temp_box_1, temp_box_2, im_width, im_height)
    
    # relative features
    output['relative_spatial_feature'] = torch.zeros(max_num_obj, max_num_obj, len(frames), 20, dtype=float)

    for f in range(len(frames)):
        for i in range( int(output['num_obj']) ):
                for j in range( int(output['num_obj']) ):
                    
                    if i<j:
                        temp_box_1 = copy(output['bboxes'])[i, f]
                        temp_box_2 = copy(output['bboxes'])[j, f]

                    # To keep the features symmetric

                    if i>=j:
                        temp_box_2 = copy(output['bboxes'])[i, f]
                        temp_box_1 = copy(output['bboxes'])[j, f]

                    output['relative_spatial_feature'][i, j, f] = relative_spatial_features(temp_box_1, temp_box_2, im_width, im_height)
    
    # video features - i3d features, motion features, others? 
    
    # i3d features
    temp_i3d_video = read_gif(file_location)
    temp_i3d_video = i3d_transform(temp_i3d_video)["video"]
    temp_i3d_video = temp_i3d_video.unsqueeze(0).to(device)
    
    temp_i3d_feature_map = i3d_feature_extractor(temp_i3d_video)
    output['i3d_feature_map'] = temp_i3d_feature_map.detach().cpu()
    
    temp_bboxes = copy(output['bboxes']).to(device)
    res_i3d_feature_map = roi_align_custom(temp_i3d_feature_map, temp_bboxes, 
                                           im_width, im_height)
    output['object_i3d_feature'] = torch.zeros(window_size, max_num_obj, 2048)
    
    for i, f in enumerate(res_i3d_feature_map):
        output['object_i3d_feature'][i] = f[:, :, 0, 0]
    
    
    # motion features
    output['motion_feature'] = torch.zeros(max_num_obj, len(frames), 5)

    for f in range(len(frames)):
        for i in range( int(output['num_obj']) ):
                    
                    temp_geom_feat_1 = output['geometric_feature'][i, f, :]
                    if f == 0:
                        temp_geom_feat_2 = 0
                    else:
                        temp_geom_feat_1 = output['geometric_feature'][i, f-1, :]
                    
                    output['motion_feature'][i, f, :] = calculate_motion_feature(temp_geom_feat_1, temp_geom_feat_2)
    
    return output



# i3d transform
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

from torchvision.transforms import Resize

i3d_transform = ApplyTransformToKey(
    key="video",
    transform=Compose(
        [
            UniformTemporalSubsample(11),
            Resize((720, 1280)),
            Lambda(lambda x: x/255.0),
            NormalizeVideo(mean, std)
        ]
    ),
)



# Loading files required to generate the trackers
# Load the annotation file
from torchvision.transforms import Resize
import pytorchvideo.models as models
import torchvision
import pickle as pkl
anno_path = '/workspace/work/misc/O2ONet/data/annotations_minus_unavailable_yt_vids.pkl'


f = open(anno_path, 'rb')
annotations = pkl.load(f)
f.close()

gif_folder = '/workspace/data/data_folder/o2o/gifs_11'

# 2d cnn feature extractor
import sys
device_num = int(sys.argv[1])


device = torch.device( 'cuda:'+ str(device_num) ) if torch.cuda.is_available() else torch.device('cpu')
deep_net = 'resnet152'
layer_no = 4
if deep_net == 'resnet152':
    model = torchvision.models.resnet152(pretrained=True)


model.to(device)
model.eval()

cnn_feature_extractor = ImageFeatureExtractor(model, layer_no, device, deep_net).to(device)

# i3d feature extractor
model_name = "i3d_r50"
model = torch.hub.load("facebookresearch/pytorchvideo:main",
                       model=model_name, pretrained=True)
model = model.to(device)
i3d_feature_net = FeatureExtractor(model, 5)




# Generating all features
from tqdm import tqdm as tqdm
import os

feature_folder = '/workspace/data/data_folder/o2o/gifs_11_features_ral_v2'
os.makedirs(feature_folder, exist_ok=True)

issues = {}
issues['index'] = []
issues['exceptions'] = []
i = 0

total = len(annotations)
total_partitions = 4

frac_init = device_num / (1.0 * total_partitions)
frac_final = (device_num + 1) / (1.0 * total_partitions)

index_init = int( total * frac_init )
index_final = int( total * frac_final )

annotations_selected = annotations[index_init: index_final]

for annotation in tqdm(annotations_selected):
    
    i+=1
    # generating location to save the feature dictionary
    yt_id = annotation['metadata']['yt_id']
    frame_index = annotation['metadata']['frame no.']
    window_size = int(int(gif_folder.split('_')[-1])/2)
    filename = yt_id + '_' + str(frame_index) + '_' + str(window_size) + '.pt'
    file_location = os.path.join(feature_folder, filename)

    if os.path.exists(file_location):
        continue
    
    # generating feature dictionary
    feature_dict = master_feature_generator(annotation, gif_folder, cnn_feature_extractor, i3d_feature_net, i3d_transform, device)
    try:
        feature_dict = master_feature_generator(annotation, gif_folder, cnn_feature_extractor, i3d_feature_net, i3d_transform, device)
    except Exception as e:
        print(" Issue in ",index_init + i)
        issues['index'].append(index_init + i)
        issues['exceptions'].append(e)
        continue
    
    # saving the feature dictionary
    torch.save(feature_dict, file_location)

import pickle as pkl
pkl.dump(issues, '/workspace/work/misc/O2ONet/feature_extraction/RAL_updates/feature_extraction/issues_dict_' + str(device_num) + '.pt')