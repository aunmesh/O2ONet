# Implements the validation
# Dependencies: loss, metrics, maybe some util files
# Called by: main.py
import loss
from utils.utils import process_data_for_fpass, process_data_for_metrics, loss_epoch_aggregator


def val( model, val_loader, config=None, metric_tracker=None):

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

    ### Creating a loss aggregator
    loss_aggregator = loss_epoch_aggregator(stage='val')

    for _, d_item in enumerate(val_loader):
        
        ### modifying data item if it needs modification before forward pass
        d_item = process_data_for_fpass(d_item, config)
        
        ### forward pass
        output_dict = model(d_item)
        
        # Getting Loss values
        loss_dict = loss.segmentation_loss(output_dict[-1], d_item['label_vector'], d_item['num_clips'])
        
        ### Adding to loss aggregator
        loss_aggregator.add(loss_dict)

        #d_item_metrics = process_data_for_metrics(d_item)
        
        ### Calculating metrics for this particular pass
        step_results = metric_tracker.calc_metrics(output_dict[-1], d_item['label_vector']
                                                    , d_item['num_clips'])
        
    ### Get the average metric for the entire run
    metric_dict = metric_tracker.aggregate_metrics()
    
    ### Get the average loss for the entire run and reset for next run
    loss_epoch = loss_aggregator.average()
    loss_aggregator.reset()
    
    ### Get the output dictionary for the validation run and reset for next run
    val_result = {**loss_epoch, **metric_dict}
    metric_tracker.reset_metrics()

    return val_result