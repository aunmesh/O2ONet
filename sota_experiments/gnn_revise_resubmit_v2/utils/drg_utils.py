import numpy as np
import torch

def bbox_trans(object_box_1_ori, object_box_2_ori, ratio, size = 64):


    object_box_1  = object_box_1_ori.clone()
    object_box_2 = object_box_2_ori.clone()
    device=object_box_1.device    
    
    InteractionPattern = [min(object_box_1[0], object_box_2[0]), 
                          min(object_box_1[1], object_box_2[1]), 
                          max(object_box_1[2], object_box_2[2]), 
                          max(object_box_1[3], object_box_2[3])]    

    height = InteractionPattern[3] - InteractionPattern[1] + 1
    width = InteractionPattern[2] - InteractionPattern[0] + 1
    
    if height > width:
        ratio = 'height'
    else:
        ratio = 'width'
        
    # shift the top-left corner to (0,0)
    
    object_box_1[0] -= InteractionPattern[0]
    object_box_1[2] -= InteractionPattern[0]
    object_box_1[1] -= InteractionPattern[1]
    object_box_1[3] -= InteractionPattern[1]    
    
    object_box_2[0] -= InteractionPattern[0]
    object_box_2[2] -= InteractionPattern[0]
    object_box_2[1] -= InteractionPattern[1]
    object_box_2[3] -= InteractionPattern[1] 

    if ratio == 'height': # height is larger than width
        
        object_box_1[0] = 0 + size * object_box_1[0] / height
        object_box_1[1] = 0 + size * object_box_1[1] / height
        object_box_1[2] = (size * width / height - 1) - size * (width  - 1 - object_box_1[2]) / height
        object_box_1[3] = (size - 1)                  - size * (height - 1 - object_box_1[3]) / height

        object_box_2[0] = 0 + size * object_box_2[0] / height
        object_box_2[1] = 0 + size * object_box_2[1] / height
        object_box_2[2] = (size * width / height - 1) - size * (width  - 1 - object_box_2[2]) / height
        object_box_2[3] = (size - 1) - size * (height - 1 - object_box_2[3]) / height
        
        # Need to shift horizontally  
        InteractionPattern = [
                                min(object_box_1[0], object_box_2[0]), 
                                min(object_box_1[1], object_box_2[1]), 
                                max(object_box_1[2], object_box_2[2]), 
                                max(object_box_1[3], object_box_2[3])]
        #assert (InteractionPattern[0] == 0) & (InteractionPattern[1] == 0) & (InteractionPattern[3] == 63) & (InteractionPattern[2] <= 63)
        if object_box_1[3] > object_box_2[3]:
            object_box_1[3] = size - 1
        else:
            object_box_2[3] = size - 1



        shift = size / 2 - (InteractionPattern[2] + 1) / 2 

        object_box_1 += torch.tensor([shift, 0 , shift, 0], device=device)
        object_box_2 += torch.tensor([shift, 0 , shift, 0], device=device)
     
    else: # width is larger than height

        object_box_1[0] = 0 + size * object_box_1[0] / width
        object_box_1[1] = 0 + size * object_box_1[1] / width
        object_box_1[2] = (size - 1)                  - size * (width  - 1 - object_box_1[2]) / width
        object_box_1[3] = (size * height / width - 1) - size * (height - 1 - object_box_1[3]) / width
        

        object_box_2[0] = 0 + size * object_box_2[0] / width
        object_box_2[1] = 0 + size * object_box_2[1] / width
        object_box_2[2] = (size - 1) - size * (width  - 1 - object_box_2[2]) / width
        object_box_2[3] = (size * height / width - 1) - size * (height - 1 - object_box_2[3]) / width
        
        # Need to shift vertically 
        InteractionPattern = [min(object_box_1[0], object_box_2[0]), 
                              min(object_box_1[1], object_box_2[1]), 
                              max(object_box_1[2], object_box_2[2]), 
                              max(object_box_1[3], object_box_2[3])]
        
        
        #assert (InteractionPattern[0] == 0) & (InteractionPattern[1] == 0) & (InteractionPattern[2] == 63) & (InteractionPattern[3] <= 63)
        

        if object_box_1[2] > object_box_2[2]:
            object_box_1[2] = size - 1
        else:
            object_box_2[2] = size - 1

        shift = size / 2 - (InteractionPattern[3] + 1) / 2 
        
        object_box_1 = object_box_1 + torch.tensor([0, shift, 0 , shift], device=device)
        object_box_2 = object_box_2 + torch.tensor([0, shift, 0 , shift], device=device)
 
    return torch.round(object_box_1), torch.round(object_box_2)


def get_sp(object_box_1, object_box_2, device):
    
    InteractionPattern = [min(object_box_1[0], object_box_2[0]), min(object_box_1[1], object_box_2[1]), 
                          max(object_box_1[2], object_box_2[2]), max(object_box_1[3], object_box_2[3])]

    height = InteractionPattern[3] - InteractionPattern[1] + 1
    width = InteractionPattern[2] - InteractionPattern[0] + 1
    
    if height > width:
        H, O = bbox_trans(object_box_1, object_box_2, 'height')
    
    else:
        H, O  = bbox_trans(object_box_1, object_box_2, 'width')
    
    pattern = torch.zeros( (2,64,64), device=device )

    pattern[0, int(H[1]):int(H[3]) + 1,int(H[0]):int(H[2]) + 1] = 1
    pattern[1, int(O[1]):int(O[3]) + 1,int(O[0]):int(O[2]) + 1] = 1

    return pattern