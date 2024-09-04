import sys
sys.path.append("/workspace/work/misc/O2ONet/sota_experiments/gnn_revise_resubmit_v3")
import os
from utils.utils import get_parser, config_loader
from train import train
from val import val
from inference import inference

from dataloader.dataset import dataset
from tqdm import tqdm as tqdm
import numpy as np
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
import torch
from log_results import wandb_logger
from metrics.metrics import metric_tracker
from utils.main_utils import *

import pickle as pkl
import os



def get_gif_files(folder_path):
    gif_files = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.gif'):
            gif_files.append(file_name)
    return gif_files



def get_indices_map(temp_dataset, gif_names):

    names_to_indices_map = {}

    for n in gif_names:
        names_to_indices_map[n] = None
    
    # data_gif_names = []
    
    for i, d in enumerate(temp_dataset):

        yt_id = d['metadata']['yt_id']
        frame_no = d['metadata']['frame no.']
        
        temp_key = yt_id + '_' + frame_no
        temp_str = temp_key + '_5.gif'
        # data_gif_names.append(temp_str)
        
        if temp_str in gif_names:
            names_to_indices_map[temp_key] = i    

    return names_to_indices_map



def format_output(predictions, target):
    '''
    predictions : list of dimension [b_size, max_num_obj_pairs, num_classes]
    target      : list of dimension [b_size, max_num_obj_pairs, num_classes]
                  they correspond with the predictions
    '''

    mask = target['num_relation'][0]

    keys = ['cr', 'lr', 'mr']

    # Maps for converting class indices to labels
    cr_map = {0: 'Contact', 1: 'No Contact', 2: 'None of these'}
    lr_map = {0: 'Below/Above', 1: 'Behind/Front', 2: 'Left/Right', 3: 'Inside', 4: 'None of these'}
    mr_map = {0: 'Holding', 1: 'Carrying', 2: 'Adjusting', 3: 'Rubbing', 4: 'Sliding', 5: 'Rotating', 
            6: 'Twisting', 7: 'Raising', 8: 'Lowering', 9: 'Penetrating', 10: 'Moving Toward', 
            11: 'Moving Away', 12: 'Negligible Relative Motion', 13: 'None of these'}

    all_predictions = {}

    mask = target['num_relation'][0]

    # Threshold for multi-label classification
    threshold = 0.5

    for k in keys:
        temp_predictions = predictions['combined'][k][0, :mask, :]
        
        # Apply the activation functions to get probabilities
        if k != 'cr':
            temp_predictions = torch.sigmoid(temp_predictions)
        else:
            temp_predictions = torch.nn.functional.softmax(temp_predictions, dim=-1)
        
        if k == 'cr':
            # Single-class prediction: argmax to get the most likely class
            predicted_classes = torch.argmax(temp_predictions, dim=-1).tolist()
            predicted_labels = [cr_map[pred] for pred in predicted_classes]
        
        else:
            # Multi-class prediction: apply threshold to get predicted classes
            predicted_classes = (temp_predictions > threshold).tolist()
            predicted_labels = []
            for preds in predicted_classes:
                labels = [label for idx, label in enumerate(lr_map.values() if k == 'lr' else mr_map.values()) if preds[idx]]
                predicted_labels.append(labels if labels else ['None of these'])
        
        # Store predictions in the all_predictions dictionary
        all_predictions[k] = predicted_labels
            
    return all_predictions

def convert_tensor_values_to_float(input_dict):
    """
    Converts all values in the dictionary that are tensors to float dtype,
    except for tensors of dtype torch.long.
    """
    # Initialize an empty dictionary to store the results
    output_dict = {}
    
    for key, value in input_dict.items():
        # Check if the value is a tensor and not of type torch.long
        if isinstance(value, torch.Tensor) and value.dtype != torch.long:
            # Convert the tensor value to float and update in the new dictionary
            output_dict[key] = value.float()
        else:
            # If the value is not a tensor or is of type torch.long, 
            # directly update in the new dictionary
            output_dict[key] = value
    
    return output_dict



def process_data_for_fpass(data_item, config):
    
    all_keys = data_item.keys()
    
    for k in all_keys:
        if isinstance(data_item[k], int):
            data_item[k] = torch.tensor([data_item[k]])
            continue
        if isinstance(data_item[k], torch.Tensor):
            data_item[k] = data_item[k].unsqueeze(0)
            continue
    
    data_item['num_relation'] = data_item['num_relation'].to(config['device'])
    data_item['num_obj'] = data_item['num_obj'].to(config['device'])

    data_item['object_pairs'] = data_item['object_pairs'].type(torch.long)
    
    # Padded features
    obj_features = []
    
    for f in config['features_list']:
        
        if f in config['custom_filter_dict'].keys():
            
            # increased by 1 to take care of the batching dimension
            frame_dim = config['custom_filter_dict'][f]['frame_dim'] + 1
            
            frame_index = config['custom_filter_dict'][f]['frame_index']
            frame_index = torch.tensor([frame_index])
            
            temp_feat = data_item[f].index_select( 
                                                    dim = frame_dim,
                                                    index = frame_index
                                                ).squeeze().to(config['device']).unsqueeze(0)
            
            obj_features.append(temp_feat)
        
        else:
            temp_feat = data_item[f].to(config['device'])
            obj_features.append(temp_feat)


    # return obj_features
    obj_features = torch.cat(obj_features, 2)

    
    
    
    # WRONG, FIX THE SLICING BY INSPECTING THE FEATURES 
    data_item['concatenated_node_features'] = obj_features.to(config['device'])[:,:,:3130]

    interaction_centric_features = []
    
    for f in config['relative_features_list']:
        temp_feat = data_item[f].flatten(3).to(config['device'])
        print(k, temp_feat.shape)
        interaction_centric_features.append(temp_feat)

    # WRONG, FIX THE SLICING BY INSPECTING THE FEATURES 
    data_item['interaction_feature'] = torch.cat(interaction_centric_features, 3)[:,:,:,:1087]
    data_item = convert_tensor_values_to_float(data_item)

    return data_item



import torch



def main(args):
    torch.set_default_dtype(torch.float)  # For Float

    ### Load Config
    config = config_loader(args.config)
    
    ### Load Model
    model = get_model(config)

    best_model_path = os.path.join(config['model_saving_path'], args.run_id, 'best_model.pth')

    f_obj = open(best_model_path,'rb')
    checkpoint = torch.load(f_obj)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    inf_features = pkl.load(open(args.inf_file_loc, 'rb'))
            
    model = model.eval()
    
    # Process the data for forward pass (assuming `process_data_for_fpass` takes the batch and config)
    d_item = process_data_for_fpass(inf_features, config)

    # Perform a forward pass through the model
    with torch.no_grad():  # Ensure no gradients are computed, since this is an inference
        output_dict = model(d_item)

    print(output_dict)
    
    final_output = format_output(output_dict, inf_features)
    print(final_output)
    
    
if __name__ == "__main__":
    
    parser = get_parser()
    args = parser.parse_args()
    main(args)