import torch
import numpy as np
import torchvision.transforms as transforms
from skimage.measure import regionprops, moments, moments_hu
import matplotlib.pyplot as plt


# Feature extraction functions
def extract_features(mask, H, W):
    # mask_np = mask.cpu().squeeze().numpy()
    mask_np = mask.squeeze()
    
    props = regionprops(mask_np.astype(int))[0]
    
    # Normalize region-based features with respect to image dimensions
    normalized_area = props.area / (H * W)
    normalized_centroid_x = props.centroid[0] / H
    normalized_centroid_y = props.centroid[1] / W
    normalized_perimeter = props.perimeter / (2*(H + W)) # normalization considering the perimeter of the image
    normalized_major_axis_length = props.major_axis_length / max(H, W)
    normalized_minor_axis_length = props.minor_axis_length / max(H, W)

    # Combining the normalized features
    region_features = [
        normalized_area,
        normalized_centroid_x,
        normalized_centroid_y,
        props.eccentricity,
        props.solidity,
        props.extent,
        normalized_perimeter,
        normalized_major_axis_length,
        normalized_minor_axis_length
    ]

    m = moments(mask_np)
    hu = moments_hu(m)

    combined_features = region_features + hu.tolist()
    return np.array(combined_features)

def extract_features_from_masks(masks, H, W):
    feature_list = []
    for mask in masks:
        features = extract_features(mask, H, W)
        feature_list.append(features)
    return np.stack(feature_list)



import torch

def get_shape_features_object_centric(feat, H, W):
    
    result = torch.zeros((12, 11, 16))

    num_obj = feat['num_obj']
    valid_masks = feat['sam_masks'][:num_obj]

    for i, v in enumerate(valid_masks):

        result[i, :, :] = torch.from_numpy( extract_features_from_masks(v, H, W) )

    feat['object_centric_shape_feats'] = result
    return feat


def get_shape_features_interaction_centric(feat):
    
    result = torch.zeros(12, 12, 11, 16)

    num_obj = feat['num_obj']
    valid_masks = feat['sam_masks'][:num_obj]

    ocf = feat['object_centric_shape_feats']

    for i in range(num_obj):
        for j in range(num_obj):

            result[i, j] = ocf[i] - ocf[j]

    feat['interaction_centric_shape_feats'] = result
    return feat