# Implements the validation
# Dependencies: loss, metrics, maybe some util files

from utils.utils import process_data_for_fpass

def inference( model, inference_set, config):

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

    d_item = process_data_for_fpass(inference_set, config)
    output_dict = model(d_item)

    return output_dict