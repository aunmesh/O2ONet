from model.nn_nets.gpnn import GPNN as GPNN
import torch
import os


from dataloader.dataset import dataset

import torch.nn as nn

from metrics.metrics import metric_tracker, metric_tracker_multi_stream, metric_tracker_action_recog


def construct_criterions(config):
    
    criterions = {}
    
    criterions['cr'] = nn.CrossEntropyLoss().to(config['device'])
    criterions['lr'] = nn.BCEWithLogitsLoss().to(config['device'])
    criterions['mr'] = nn.BCEWithLogitsLoss().to(config['device'])
    
    return criterions


def get_metric_trackers(config):
    
    if config['model_name'] == 'action_recog_test':
        train_metric_tracker = metric_tracker_action_recog(config, mode='train')
        val_metric_tracker = metric_tracker_action_recog(config, mode='val')
        test_metric_tracker = metric_tracker_action_recog(config, mode='test')
        
        return train_metric_tracker, val_metric_tracker, test_metric_tracker
    
    else:

        train_metric_tracker = metric_tracker_multi_stream(config, mode='train')
        val_metric_tracker = metric_tracker_multi_stream(config, mode='val')
        test_metric_tracker = metric_tracker_multi_stream(config, mode='test')

        return train_metric_tracker, val_metric_tracker, test_metric_tracker
        


from model.nn_nets.imp import imp
from model.nn_nets.squat import SQUAT
from model.nn_nets.mlp_baseline import MLP

def get_model(config):
    model_map = {
        'imp': imp,
        'GPNN': GPNN,
        'SQUAT': SQUAT,
        'MLP': MLP
    }

    model_class = model_map.get(config['model_name'])
    if model_class:
        return model_class(config).to(config['device'])
    else:
        raise ValueError(f"Unknown model name: {config['model_name']}")


def get_dataset(config, split='train'):

    return dataset(config, split)


def freeze_layers(model, freezing_list):

    named_params = [p for p in model.named_parameters()]

    for temp_param in named_params:
        if temp_param[0] in freezing_list:
            temp_param[1].requires_grad=False

def unfreeze_model(model):

    named_params = [p for p in model.named_parameters()]

    for temp_param in named_params:
        temp_param[1].requires_grad=True
    

def get_optimizer(config, model):

    if config['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    return optimizer

def load_checkpoint(config, model, optimizer):

    checkpoint_path = os.path.join(config['model_saving_path'], config['run_id'], 'checkpoint.pth')

    try:
        f_obj = open(checkpoint_path,'rb')
        checkpoint = torch.load(f_obj)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return model, optimizer, start_epoch
    
    except:
        return model, optimizer, 0


def save_state(model, optimizer, epoch, config, best=False):

    os.makedirs(os.path.join(config['model_saving_path'], config['run_id']), exist_ok=True)
    checkpoint_path = os.path.join(config['model_saving_path'], config['run_id'], 'checkpoint.pth')
    best_model_path = os.path.join(config['model_saving_path'], config['run_id'], 'best_model.pth')

    checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict()
    }

    torch.save(checkpoint, checkpoint_path)

    if best:
        torch.save(checkpoint, best_model_path)