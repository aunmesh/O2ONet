# Implements the validation
# Dependencies: loss, metrics, maybe some util files

from utils.utils import process_data_for_fpass, process_data_for_metrics, loss_aggregator

def test( model, test_loader, config, metric_tracker):
    '''
    args:
        model   :
        test_loader   : 
        config  :
    returns:
    '''
    # Forward pass on the test_loader
    # Collect the metrics for each batch
    # Average the metrics
    # return a result_dict
    
    for _, d_item in enumerate(test_loader):
        
        d_item = process_data_for_fpass(d_item, config)
        output_dict = model(d_item)
        
        test_step_result = metric_tracker.calc_metrics(output_dict[-1], d_item['label_vector']
                                                   , d_item['num_clips'])
        
    test_result = metric_tracker.aggregate_metrics()
    metric_tracker.reset_metrics()

    return test_result