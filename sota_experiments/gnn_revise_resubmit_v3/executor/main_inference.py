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
from utils.utils import process_data_for_fpass


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

def main(args):
    torch.set_default_dtype(torch.float)  # For Float

    ### Load Config
    config = config_loader(args.config)
    
    ### Load Model
    model = get_model(config)
    
    # if config['log_results']:
    #     logger = wandb_logger(config)
    # else:
    #     pass

    best_model_path = os.path.join(config['model_saving_path'], args.run_id, 'best_model.pth')

    f_obj = open(best_model_path,'rb')
    checkpoint = torch.load(f_obj)
    model.load_state_dict(checkpoint['model_state_dict'])

    # if config['log_results']:
    #     config = logger.config

    ### Creating the test data loader
    full_dataset__ = get_dataset(config, 'inference')
    l_ = len(full_dataset__)


    full_dataset = [full_dataset__[i][0] for i in range(l_)]
    del full_dataset__    

    # How are we getting the inference set now. Now we are getting it using the 
    # How do we want to define the inference set eventually. Eventually we just want to put the inference gifs inside a folder and the whole process of inference
    # should happen.
    
    # How are we doing it now?
        # Now we should just mimic that. Put the gif in a folder. Using that we extract a particular feature_dict and then send that to model.

    '''
        1. Get the name of all the gifs in the gif folder
        2. Use the names to get the corresponding indices in the dataset
        3. Once the indices are there, get the data_item
    ''' 
    gif_folder_loc = args.inference_folder_location
    
    gif_files = get_gif_files(gif_folder_loc)
    print("FLAG 95", gif_files)
    gif_to_ind_map = get_indices_map(full_dataset, gif_files)
    print("FLAG 97", gif_to_ind_map)
        
    indices_available = []
    for k in gif_to_ind_map.keys():
        if gif_to_ind_map[k] != None:
            indices_available.append(gif_to_ind_map[k])
    

    from torch.utils.data import DataLoader, Subset

    # # Custom collate function to combine dictionary items in a batch
    # def collate_fn(batch):
    #     return {key: torch.stack([item[key] for item in batch]) for key in batch[0].keys()}

    # Create a subset of the dataset using the indices
    subset = Subset(full_dataset, indices_available)
    print("FLAG SPEC", type(subset[0]))

    dataloader = DataLoader(subset, batch_size=len(indices_available), shuffle=False)
    
    model = model.eval()

    # Directly get the single batch from the dataloader
    inference_set = next(iter(dataloader))

    # Process the data for forward pass (assuming `process_data_for_fpass` takes the batch and config)
    d_item = process_data_for_fpass(inference_set, config)

    # Perform a forward pass through the model
    with torch.no_grad():  # Ensure no gradients are computed, since this is an inference
        output_dict = model(d_item)

    print(output_dict)
    
    
    # ### Logging test result
    # if config['log_results']:
    #     logger.log_dict(test_result)

if __name__ == "__main__":
    
    parser = get_parser()
    args = parser.parse_args()
    main(args)