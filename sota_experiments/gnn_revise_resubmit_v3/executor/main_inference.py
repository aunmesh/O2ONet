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



def format_output(predictions, target):
    '''
    predictions : list of dimension [b_size, max_num_obj_pairs, num_classes]
    target      : list of dimension [b_size, max_num_obj_pairs, num_classes]
                  they correspond with the predictions
    '''

    mask = target['num_relation']

    keys = ['cr', 'lr', 'mr']

    b_size = target['lr'].shape[0]
    tot_num_rels = 0
    
    cr_map = {'Contact': 0, 'No Contact': 1, 'None of these': 2, '': 2}
    lr_map = {'Below/Above': 0, 'Behind/Front': 1, 'Left/Right': 2, 'Inside': 3, 'None of these': 4, '': 4}
    mr_map = {'Holding': 0, 'Carrying': 1, 'Adjusting': 2, 'Rubbing': 3, 'Sliding': 4, 'Rotating': 5, 'Twisting': 6,
              'Raising': 7, 'Lowering': 8, 'Penetrating': 9, 'Moving Toward': 10, 'Moving Away': 11, 
              'Negligible Relative Motion': 12, 'None of these': 13, '': 13}
        
    all_predictions = {}

    for b in range(b_size):
        
        curr_num_rel = int(mask[b])
        tot_num_rels+=curr_num_rel
        
        temp_predictions = {}
        predicted_rels = []
        for k in keys:
            temp_predictions = predictions['combined'][k][b, :curr_num_rel, :]
            
            # Apply the squashing functions appropriately to get the probabilities
            if k!='cr':
                temp_predictions = torch.sigmoid(temp_predictions)
            
            if k=='cr':
                temp_predictions = torch.nn.functional.softmax(temp_predictions, dim=-1)
    
    # We want to convert the results into the text
    
    # How is the whole thing working. Are we stacking up the tensors as we need them?
    # From the GNN, Are we stacking up the tensors, as they are needed?
    # How are they being stacked? They are being stacked according to the ground truth. More specifically they
    # are being stacked according to the pairs matrix. So we also need the pairs matrix, to realize how they were stacked.
    
    # So, what is the output. One we can convert this tensor to the words. But then we will have to assign the words to the concerned object
    # pair and then display that.
    
    # It is not a complex task and can be done fast.
    
    # So, 1st just convert the detections to words.
    
    # So, what will the output look like:
    # For each pair, we can have a dictionary which classifies all the outputs.
    
    # Then depending on the number of pairs annotated there will be a list of dictionaries for each gif in a batch.
    # Then depending on the batch size there will be a list of lists
    # So finally it is a list of lists of dictionary [[{}, {}]]
            
            
    
    
    
    
    return loss



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