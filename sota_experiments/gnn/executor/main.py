import sys
sys.path.append("/workspace/work/O2ONet/sota_experiments/gnn")
import os
from utils.utils import get_parser, config_loader
from train import train
from val import val
from test import test
from dataloader.dataset import dataset
from tqdm import tqdm as tqdm
import numpy as np

from torch.utils.data import Dataset, DataLoader
import torch
from log_results import wandb_logger
from metrics.metrics import metric_tracker
from utils.main_utils import *

def main(args):
    
    ### Load Config
    config = config_loader(args.config)
    
    ### Load Model
    model = get_model(config)
    
    ### Load Optimizer
    optimizer = get_optimizer(config, model)
    
    ### Load Logger  ( UNCOMMENT ) 
    # logger = wandb_logger(config)

    start_epoch = 0

    ### Check if resuming training (Commented)

    config['run_id'] = 'debug'
    
    # if args.resume:
    #     config['run_id'] = args.run_id
    #     model, optimizer, start_epoch = load_checkpoint(config, model, optimizer)
    # config = logger.config

    ### Creating metric tracker objects
    train_metric_tracker = metric_tracker(config, mode='train')
    val_metric_tracker = metric_tracker(config, mode='val')
    test_metric_tracker = metric_tracker(config, mode='test')
    
    ### Training 
    if args.train:
        
        ### create training data loader        
        train_dataset = get_dataset(config, 'train')
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'])

        ### create validation data loader
        val_dataset = get_dataset(config, 'val')
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])

        end_epoch = config['num_epochs']

        ### For saving the best model (to report the test performance) using validation mAP 
        best_mAP = -np.inf
        
        ### Training Loop
        # for e in tqdm(range(start_epoch, end_epoch)):
        for e in tqdm(range(start_epoch, 1)):

            ### Train for an epoch and get the result dictionary
            train_result = train(model, train_loader, optimizer, config, train_metric_tracker)
            
            ### Validation for an epoch and get the result dictionary
            val_result = val(model, val_loader, config, val_metric_tracker)
            
            ### Save the model and log results
            is_best = False
            if best_mAP < val_result['val_mAP_all']:
                best_mAP = val_result['val_mAP_all']
                is_best = True

            ### Saving the best model if it is there
            save_state(model, optimizer, e, config, is_best)  # ADD FUNCTIONALITY FOR SAVING BEST MODEL
            
            ### logging the training and validation result for viewing
            # logger.log_dict({**train_result, **val_result})

        ### Testing the model
        
        ### Creating the test data loader
        test_dataset = get_dataset(config, 'test')
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size_test'])
        
        ### Obtaining test result
        test_result = test(model, test_loader, config, test_metric_tracker )
        
        ### Logging test result
        # logger.log_dict(test_result)

if __name__ == "__main__":
    
    parser = get_parser()
    args = parser.parse_args()
    main(args)