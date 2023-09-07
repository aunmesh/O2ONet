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
from log_results import CrossValidationLogger
from metrics.metrics import metric_tracker
from utils.main_utils import *

from utils.stratify import get_stratified_splits_indices, CrossValidationAggregator
from utils.stratify import modify_keys_for_fold

def main(args):
    
    ### Load Config
    config = config_loader(args.config)
 
    n_folds = 5
    config['num_folds'] = n_folds
    config['master_log_dir'] = './logs'
    config['split_file_id'] = args.stratified
    
    logger = CrossValidationLogger(config)

    start_epoch = 0

    ### Creating metric tracker objects
    train_metric_tracker, val_metric_tracker, test_metric_tracker = get_metric_trackers(config)

    cv_train_aggregator = CrossValidationAggregator()
    cv_val_aggregator = CrossValidationAggregator()

    ### Training 
    if args.train:
        
        ### create training data loader        
        full_dataset = get_dataset(config, 'full')

        end_epoch = config['num_epochs']

        ### For saving the best model (to report the test performance) using validation mAP 
        best_mAP = -np.inf

        import pickle
        
        f_ptr = open('./stratified_splits_' + str(config['split_file_id']) + '.pkl', 'rb')
        splits = pickle.load(f_ptr)
        f_ptr.close()        
        
        
        for fold in range(n_folds):
                        
            ### Load Model
            model = get_model(config)
            
            ### Load Optimizer
            optimizer = get_optimizer(config, model)
            
            ### Construct criterions
            criterions = construct_criterions(config)


            print("In Validation Loop ", fold)
            
            train_indices = splits[0][fold]
            val_indices = splits[1][fold]
            
            logger.log_fold_start(fold + 1, train_indices, val_indices)
            
            # Set indices for the train and validation subsets
            full_dataset.set_indices(train_indices)
            train_loader = DataLoader(full_dataset, batch_size=config['train_batch_size'], 
                                      shuffle=True, drop_last=True)
            
        
        
            ### Training Loop
            for e in tqdm(range(start_epoch, end_epoch)):

                ### Train for an epoch and get the result dictionary
                train_result = train(model, train_loader, optimizer, config, criterions, train_metric_tracker)
                

                cv_train_aggregator.add_data(fold=fold, epoch=e, 
                                             data=train_result)
                
                # Similar step for validation set:
                full_dataset.set_indices(val_indices)
                val_loader = DataLoader(full_dataset, batch_size=config['val_batch_size'])

                ### Validation for an epoch and get the result dictionary
                val_result = val(model, val_loader, config, criterions, val_metric_tracker)

                cv_val_aggregator.add_data(fold=fold, epoch=e, 
                                             data=val_result)

                
                new_train_result = modify_keys_for_fold(train_result, fold)
                new_val_result = modify_keys_for_fold(val_result, fold)
                
                epoch_results = {**new_train_result, **new_val_result}
                logger.log_fold_metrics(fold + 1, epoch_results)
            
            del model
            del optimizer
            del criterions

            logger.log_fold_end(fold + 1)

        agg_val_result = cv_val_aggregator.aggregate()
        agg_train_result = cv_train_aggregator.aggregate()
        agg_results = {**agg_train_result, **agg_val_result}
        logger.log_summary_metrics(agg_results)
        logger.save()

if __name__ == "__main__":
    
    parser = get_parser()
    args = parser.parse_args()
    main(args)