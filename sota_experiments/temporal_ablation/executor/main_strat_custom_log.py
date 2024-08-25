import sys
sys.path.append("/workspace/work/misc/O2ONet/sota_experiments/temporal_ablation/")
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


tensor_dims = {
    "relative_spatial_feature": 20,
}


# object_i3d_feature torch.Size([5, 12, 2048])
# bbox_CLIP torch.Size([12, 768])
# geometric_feature torch.Size([12, 11, 5])
# object_semantic_embeddings torch.Size([12, 300])
# object_centric_shape_feats torch.Size([12, 11, 9])
    
# relative_spatial_feature torch.Size([12, 12, 11, 20])
# interaction_bbox_CLIP torch.Size([12, 12, 768])
# interaction_centric_shape_feats torch.Size([12, 12, 11, 9])


# CLIP, i3d, shape, word2vec, bbox_geometric 

def main(args):
    
    ### Load Config
    config = config_loader(args.config)
    config['device'] = torch.device("cuda:" + str(args.gpu))
    
    config['num_frames'] = int(args.num_frames)
    
    shift_rel_spa = 20 * (11 - config['num_frames'])
    shift_int_cent_sha = 9 * (11 - config['num_frames'])
    config['edge_feature_size'] -= (shift_rel_spa + shift_int_cent_sha)
    # shift_geom = 5 * (11 - config['num_frames'])
    # shift_obj_cent = 9 * (11 - config['num_frames'])
    # config['node_feature_size'] -= (shift_geom + shift_obj_cent)

    config['ablated_model_name'] = config['model_name'] + '_' + str(config['num_frames'])
    
    n_folds = 5
    config['num_folds'] = n_folds
    config['master_log_dir'] = './logs'
    config['split_file_id'] = args.stratified
    

    logger = CrossValidationLogger(config)
    logger.load()  # Load the previous data

    last_fold = logger.last_completed_fold()

    if last_fold is not None:
        start_fold = last_fold + 1
    else:
        start_fold = 1    

    start_epoch = 0

    ### Creating metric tracker objects
    train_metric_tracker, val_metric_tracker, test_metric_tracker = get_metric_trackers(config)

    cv_train_aggregator = CrossValidationAggregator()
    cv_val_aggregator = CrossValidationAggregator()

    ### Load Model
    model = get_model(config)
    
    ### Load Optimizer
    optimizer = get_optimizer(config, model)
    
    ### Construct criterions
    criterions = construct_criterions(config)

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
        
        
        for fold in range(start_fold -1, n_folds):
                        


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

            print("FLAG 1")
            logger.log_fold_end(fold + 1)
            logger.save()
            

        logger.aggregate()
        logger.save()

if __name__ == "__main__":
    
    parser = get_parser()
    args = parser.parse_args()
    main(args)