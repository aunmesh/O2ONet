import torch
import torchvision.models as models
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np

def get_roi_class_probabilities(image, boxes, model, preprocess):
    class_probs = []
    
    device = next(model.parameters()).device

    for box in boxes:
        
        temp_box = np.array(box)
        roi = image.crop(temp_box)
        
        # Preprocess the RoI
        roi_tensor = preprocess(roi).unsqueeze(0).to(device)
        
        with torch.no_grad():
            temp_probs = model(roi_tensor)
            probs = torch.nn.functional.softmax(temp_probs, dim=1)
            class_probs.append(probs)
    
    return torch.cat(class_probs)

def compute_weighted_embeddings(class_probs, embeddings_tensor):
    # Compute the weighted sum of embeddings using class probabilities
    weighted_embeddings = torch.mm(class_probs, embeddings_tensor)
    return weighted_embeddings


def get_semantic_embeddings(ml_elements, gif, feat, hand_indices):
    """
    ml_elements['clip']         : clip model (modified by ViPLO) to be used for extracting features
    ml_elements['clip_preproc'] : pre-processing function to be used to pre-process the features
    video                       : Object having the video
    feat                        : The current extracted feats also having the tracked bounding boxes
    """

    num_obj = int(feat['num_obj'])
    
    bbox_tensors = feat['bboxes'][:num_obj, :]
    
    obj_indices = []
    for i in range(num_obj):
        if i not in hand_indices:
            obj_indices.append(i)
    
    obj_bboxes = bbox_tensors[obj_indices, 5, :].tolist()
    
    frame = gif[5]
    
    roi_det_model = ml_elements['ROI_DET_MODEL']
    roi_preprocess = ml_elements['ROI_PREPROCESS']
    embeddings_tensor = ml_elements['imagenet_word2vec_embed']
    
    temp_class_probs = get_roi_class_probabilities(frame, obj_bboxes,
                                roi_det_model, roi_preprocess)
    
    temp_semantic_embeddings = compute_weighted_embeddings(temp_class_probs,
                                                           embeddings_tensor)
    
    semantic_embeddings = torch.zeros((12, 300), device=temp_semantic_embeddings.device)
    
    semantic_embeddings[obj_indices, :] = temp_semantic_embeddings
    semantic_embeddings[hand_indices, :] = ml_elements['hand_embedding']
    
    feat['object_semantic_embeddings'] = semantic_embeddings
    
    return feat