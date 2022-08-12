# Implements the training code
# Dependencies: loss.py, probably some utils, metrics,
# Called by main.py

import torch
import loss
from utils.utils import process_data_for_fpass, process_data_for_metrics, loss_epoch_aggregator
from tqdm import tqdm as tqdm

def train(model, train_loader, optimizer, config, metric_tracker):
    """
    Args:
        model     :
        train_loader:
        optimizer   : 
        config    :
        metric_tracker:
    
    Returns:
        train_result:
    """
    
    ### For aggregating the loss across various epochs
    loss_aggregator = loss_epoch_aggregator(stage='train')

    ### Trains for One Epoch
    for _, d_item in tqdm(enumerate(train_loader)):

        ### Re-setting optimizer
        optimizer.zero_grad()
        
        ### Modifying the data item if it needs modification before training
        d_item = process_data_for_fpass(d_item, config)

        ### Forward Pass
        output_dict = model(d_item)

        ### Loss Calculation
        loss_dict = loss.segmentation_loss(output_dict[-1], d_item['label_vector'], d_item['num_clips'])
        
        ### Backward Pass
        loss_dict['loss_total'].backward()

        ### Loss aggregation for this epoch
        loss_aggregator.add(loss_dict)
        
        ### Processing data for metric calculation if it needs processing
        # d_item_metrics = process_data_for_metrics(d_item)
        
        ### Calculating metrics for this iteration
        step_results = metric_tracker.calc_metrics(output_dict[-1], d_item['label_vector']
                                                   , d_item['num_clips'])
        
        ### Optimizing the Model
        optimizer.step()
    
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