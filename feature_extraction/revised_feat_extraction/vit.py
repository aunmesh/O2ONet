import torch

def resize_bboxes(bboxes, original_dim, new_dim):
    """
    Rescale bounding box coordinates based on new image dimensions and round to integers.

    Args:
    - bboxes (torch.Tensor): Tensor of shape [num_bboxes, 4].
                             Each bounding box is represented as [xmin, ymin, xmax, ymax].
    - original_dim (tuple): Original dimensions as (original_width, original_height).
    - new_dim (tuple): New dimensions as (W, H).

    Returns:
    - torch.Tensor: Rescaled bounding boxes with integer values.
    """
    # Clone the tensor to ensure the original tensor is not modified
    bboxes_rescaled = bboxes.clone()

    original_width, original_height = original_dim
    W, H = new_dim

    # Calculate scaling factors for width and height
    width_scale = W / original_width
    height_scale = H / original_height

    # Apply scaling and rounding to the cloned tensor
    bboxes_rescaled[:, [0, 2]] = torch.round(bboxes_rescaled[:, [0, 2]] * width_scale)
    bboxes_rescaled[:, [1, 3]] = torch.round(bboxes_rescaled[:, [1, 3]] * height_scale)

    return bboxes_rescaled



def extract_vit_feats(ml_elements, gif, feat):
    """
    ml_elements['clip']         : clip model (modified by ViPLO) to be used for extracting features
    ml_elements['clip_preproc'] : pre-processing function to be used to pre-process the features
    video                       : Object having the video
    feat                        : The current extracted feats also having the tracked bounding boxes
    """
    
    # Flow - Get the bounding boxes, Get the image into the format needed, get it pre-processed
    # Get the bounding box features, Return it as a feature matrix
    
    num_obj = int(feat['num_obj'])
    
    img_width, img_height = feat['metadata']['frame_width'], feat['metadata']['frame_height']

    img_dim = (img_width, img_height)
    new_dim = (224, 224)
            
    bbox_tensors = feat['bboxes'][:num_obj, 5]
    bbox_tensors_scaled = [resize_bboxes(bbox_tensors, img_dim, new_dim)]
    
    central_frame = gif[5]
    central_frame_tensor = ml_elements['VIPLO_PRE_PROC'](central_frame).unsqueeze(0).to(ml_elements['device'])
    with torch.no_grad():
        bbox_feature, image_features = ml_elements['VIPLO_CLIP'].encode_image(central_frame_tensor, 
                                                                bbox_tensors_scaled)
    

    expanded_tensor = torch.cat((bbox_feature, torch.zeros(12-num_obj,
                                768, device=ml_elements['device'])), dim=0)
    
    feat['bbox_CLIP'] = expanded_tensor
    
    return feat