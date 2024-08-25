import numpy as np
import cv2

def rescale_masks(mask):

    # First, ensure mask isn't None
    if mask is not None:
        # Squeeze out the first dimension
        mask_squeezed = mask.squeeze().astype(np.uint8)
        
        # Ensure mask values are 0 or 1 (binary). This is just a precaution.
        mask_squeezed = np.where(mask_squeezed > 0, 1, 0).astype(np.uint8)

        # Rescale the mask to 224x224
        rescaled_mask = cv2.resize(mask_squeezed, (320, 240), interpolation=cv2.INTER_NEAREST)
        mask_bool = rescaled_mask.astype(bool)
        return mask_bool
    
    else:
        return None


def extract_sam_masks(ml_elements, gif, feat):
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
    bbox_tensors = feat['bboxes'][:num_obj, :]
    
    # sam_masks = torch.zeros(12, 11, 320, 240, dtype=torch.bool)
    
    sam_masks = np.zeros((12, 11, 240, 320) ,dtype=bool)
    sam_predictor = ml_elements['SAM_predictor']

    for i in range(11):
        temp_gif = gif[i].convert('RGB')
        frame = np.asarray(temp_gif)
               
        sam_predictor.set_image(frame)
        
        for j in range(num_obj):
            
            temp_box = bbox_tensors[j, i].numpy()
            
            masks, _, _ = sam_predictor.predict(
                                                    point_coords=None,
                                                    point_labels=None,
                                                    box=temp_box,
                                                    multimask_output=False,
                                                )
            rescaled_masks = rescale_masks(masks)
            
            if type(rescaled_masks) is not None:
                sam_masks[j, i, :, :] = rescaled_masks
    
    feat['sam_masks'] = sam_masks

    return feat