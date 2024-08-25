import sys
sys.path.append("/workspace/work/misc/O2ONet/sota_experiments/revise_ablation")
import os
from utils.utils import get_parser, config_loader
from train import train
from val import val
from test import test
from dataloader.dataset import dataset
from tqdm import tqdm as tqdm
import numpy as np
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
import torch
from log_results import wandb_logger
from metrics.metrics import metric_tracker
from utils.main_utils import *



def calculate_average(list_of_dicts):
    # Initialize a dictionary to store the sum of values for each key.
    sum_dict = {}
    count_dict = {}

    for d in list_of_dicts:
        for key, value in d.items():
            # If the key isn't in the sum_dict, add it with the current value.
            if key not in sum_dict:
                sum_dict[key] = value
                count_dict[key] = 1
            else:
                # Otherwise, just increment the existing value in sum_dict.
                sum_dict[key] += value
                count_dict[key] += 1

    # Calculate average for each key.
    avg_dict = {}
    for key in sum_dict:
        avg_dict["avg_" + key] = sum_dict[key] / count_dict[key]

    return avg_dict



def main(args):
    
    ### Load Config
    config = config_loader(args.config)
    config['ablated_feat'] = args.ablated_feat
    ### Load Model
    model = get_model(config)
    
    ### Construct criterions
    criterions = construct_criterions(config)
 
    logger = wandb_logger(config)

    start_epoch = 0

    model = load_checkpoint_2(model)
    
    train_metric_tracker, val_metric_tracker, test_metric_tracker = get_metric_trackers(config)

    val_dataset = get_dataset(config, 'full')

    import pickle
    
    f_ptr = open('./stratified_splits_' + str('0') + '.pkl', 'rb')
    splits = pickle.load(f_ptr)
    f_ptr.close()        
    
    val_indices = splits[1][0]
    
    # Set indices for the train and validation subsets
    # Similar step for validation set:
    val_dataset.set_indices(val_indices)
    val_loader = DataLoader(val_dataset, batch_size=config['val_batch_size'])

    val_results = []
    ### Training Loop
    for e in tqdm(range(0, 5)):
        
        ### Validation for an epoch and get the result dictionary
        val_result = val(model, val_loader, config, criterions, val_metric_tracker)
        val_results.append(val_result)
        logger.log_dict(val_result)
    
    result_dict = calculate_average(val_results)
    logger.log_dict(result_dict)
    
    


if __name__ == "__main__":
    
    parser = get_parser()
    args = parser.parse_args()
    main(args)