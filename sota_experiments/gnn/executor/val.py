# Implements the validation
# Dependencies: loss, metrics, maybe some util files
# Called by: main.py
import loss
from utils.utils import process_data_for_fpass, process_data_for_metrics, loss_epoch_aggregator, get_loss


def val( model, val_loader, config, criterions, metric_tracker):

    '''
    args:
        model   :
        val_loader   : 
        config  :
    returns:
    '''

    # Forward pass on the val_loader
    # Collect the metrics for each batch
    # Average the metrics
    # return a result_dict
    ### set model to eval mode
    model = model.eval()
   
    ### Creating a loss aggregator
    loss_aggregator = loss_epoch_aggregator(stage='val')

    for _, data_item in enumerate(val_loader):
        
        d_item, idx = data_item
        
        ### modifying data item if it needs modification before forward pass
        d_item = process_data_for_fpass(d_item, config)
        
        ### forward pass
        output_dict = model(d_item)
        
        # Getting Loss values
        loss_dict = get_loss(output_dict, d_item, criterions, config)
        
        ### Adding to loss aggregator
        loss_aggregator.add(loss_dict)

        #d_item_metrics = process_data_for_metrics(d_item)
        
        ### Calculating metrics for this particular pass
        step_results = metric_tracker.calc_metrics(output_dict, d_item)
        
    ### Get the average metric for the entire run
    metric_dict = metric_tracker.aggregate_metrics()
    
    ### Get the average loss for the entire run and reset for next run
    loss_epoch = loss_aggregator.average()
    loss_aggregator.reset()
    
    ### Get the output dictionary for the validation run and reset for next run
    val_result = {**loss_epoch, **metric_dict}
    metric_tracker.reset_metrics()
    model = model.train()
    return val_result