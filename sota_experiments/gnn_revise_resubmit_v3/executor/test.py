# Implements the validation
# Dependencies: loss, metrics, maybe some util files

from utils.utils import process_data_for_fpass, process_data_for_metrics

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
    
    # set model to test mode
    model = model.eval()
    
    for _, data_item in enumerate(test_loader):
        
        d_item, idx = data_item
        
        d_item = process_data_for_fpass(d_item, config)
        output_dict = model(d_item)
        
        test_step_result = metric_tracker.calc_metrics(output_dict, d_item)
        
    test_result = metric_tracker.aggregate_metrics()
    metric_tracker.reset_metrics()

    # set model back to train mode
    model = model.train()

    return test_result