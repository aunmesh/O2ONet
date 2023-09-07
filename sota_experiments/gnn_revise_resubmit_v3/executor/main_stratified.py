import sys
sys.path.append("/workspace/work/misc/O2ONet/sota_experiments/gnn_revise_resubmit_v2")
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

from utils.stratify import get_stratified_splits_indices, CrossValidationAggregator
from utils.stratify import modify_keys_for_fold

def main(args):
    
    ### Load Config
    config = config_loader(args.config)
    
    ### Load Model
    model = get_model(config)
    
    ### Load Optimizer
    optimizer = get_optimizer(config, model)
    
    ### Construct criterions
    criterions = construct_criterions(config)
 
    ### Load Logger  ( UNCOMMENT )
    if config['log_results']:
        logger = wandb_logger(config)
    else:
        pass

    start_epoch = 0

    ### Check if resuming training (Commented)

    if not config['log_results']:
        config['run_id'] = 'debug'
    
    if args.resume:
        config['run_id'] = args.run_id
        model, optimizer, start_epoch = load_checkpoint(config, model, optimizer)
    
    if config['log_results']:
        config = logger.config

    ### Creating metric tracker objects
    train_metric_tracker, val_metric_tracker, test_metric_tracker = get_metric_trackers(config)



    # Usage:
    cv_train_aggregator = CrossValidationAggregator()
    cv_val_aggregator = CrossValidationAggregator()


    ### Training 
    if args.train:
        
        ### create training data loader        
        train_dataset = get_dataset(config, 'full')
        val_dataset = get_dataset(config, 'full')

        end_epoch = config['num_epochs']

        ### For saving the best model (to report the test performance) using validation mAP 
        best_mAP = -np.inf


        n_splits = 5
        import pickle
        
        f_ptr = open('./stratified_splits_' + str(args.stratified) + '.pkl', 'rb')
        splits = pickle.load(f_ptr)
        f_ptr.close()        
        
        
        for split in range(n_splits):
            print("In Validation Loop ", split)
            
            train_indices = splits[0][split]
            val_indices = splits[1][split]
            
            # Set indices for the train and validation subsets
            train_dataset.set_indices(train_indices)
            train_loader = DataLoader(train_dataset, batch_size=config['train_batch_size'], 
                                      shuffle=True, drop_last=True)
            
            # Similar step for validation set:
            val_dataset.set_indices(val_indices)
            val_loader = DataLoader(val_dataset, batch_size=config['val_batch_size'])
        
        
            ### Training Loop
            for e in tqdm(range(start_epoch, end_epoch)):

                ### Train for an epoch and get the result dictionary
                train_result = train(model, train_loader, optimizer, config, criterions, train_metric_tracker)
                

                cv_train_aggregator.add_data(fold=split, epoch=e, 
                                             data=train_result)
                
                ### Validation for an epoch and get the result dictionary
                val_result = val(model, val_loader, config, criterions, val_metric_tracker)

                cv_val_aggregator.add_data(fold=split, epoch=e, 
                                             data=val_result)

                
                ### logging the training and validation result for viewing
                if config['log_results']:
                    new_train_result = modify_keys_for_fold(train_result, split)
                    new_val_result = modify_keys_for_fold(val_result, split)
                    
                    logger.log_dict({**new_train_result, **new_val_result})

        agg_val_result = cv_val_aggregator.aggregate()
        agg_train_result = cv_train_aggregator.aggregate()
        
        if config['log_results']:
            logger.log_dict({**agg_train_result, **agg_val_result})

if __name__ == "__main__":
    
    parser = get_parser()
    args = parser.parse_args()
    main(args)