# Implements the training code
# Dependencies: loss.py, probably some utils, metrics,
# Called by main.py

import torch
import loss

from utils.utils import process_data_for_fpass, process_data_for_metrics
from utils.utils import loss_epoch_aggregator, get_loss

from tqdm import tqdm as tqdm
from time import time

def train(model, train_loader, optimizer, config, criterions, metric_tracker):
    """
    Args:
        model       :
        train_loader:
        optimizer   : 
        config      :
        metric_tracker:
    
    Returns:
        train_result:
    """
    ### set model to train model
    model = model.train()
    
    ### For aggregating the loss across various epochs
    loss_aggregator = loss_epoch_aggregator(stage='train')

    ### Trains for One Epoch
    for temp_flag, data_item in enumerate(train_loader):
        t_i = time()
       
        d_item, idx = data_item
        
        ### Re-setting optimizer
        optimizer.zero_grad()
        
        ### Modifying the data item if it needs modification before training
        d_item = process_data_for_fpass(d_item, config)
        
        ### Forward Pass
        output_dict = model(d_item)

        ### Loss Calculation
        loss_dict = get_loss(output_dict, d_item, criterions, config)
        
        ### Backward Pass
        loss_dict['loss_total'].backward()

        ### Loss aggregation for this epoch
        loss_aggregator.add(loss_dict)
        
        ### Calculating metrics for this iteration
        step_results = metric_tracker.calc_metrics(output_dict, d_item)

        ### Optimizing the Model
        optimizer.step()
        
        t_f = time()
        # print("Time Taken", t_f - t_i)
        
    ### Aggregating metrics across all iterations
    metric_dict = metric_tracker.aggregate_metrics()
    
    ### Resetting metric tracker for next iteration
    metric_tracker.reset_metrics()
    
    ### Calculation of average loss for the epoch (this is required otherwise in logging there will be too many datapoints)
    loss_epoch = loss_aggregator.average()

    ### Reset the loss aggregator for next run
    loss_aggregator.reset()

    ### Prepare training result (using the output from the metric calculation and loss aggregator)
    train_result = {**loss_epoch, **metric_dict}
    
    return train_result