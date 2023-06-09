from model.nn_nets.ooi_net import ooi_net as ooi_net
from model.nn_nets.gpnn import GPNN as GPNN
import torch
import os


from dataloader.dataset import dataset
from dataloader.vsgnet_dataset import vsgnet_dataset
from dataloader.gpnn_dataset import gpnn_dataset
from dataloader.action_recog_dataset import action_recog_dataset

from model.nn_nets.vsgnet import vsgnet
from model.nn_nets.drg import DRG
from model.nn_nets.ican import iCAN
from model.nn_nets.action_recog_nets.action_recog import action_net_cnn_stream
import torch.nn as nn

from metrics.metrics import metric_tracker, metric_tracker_multi_stream, metric_tracker_action_recog

def construct_criterions(config):
    
    if config['model_name'] == 'action_recog_test':
        criterions = {}
        criterions['action'] = nn.CrossEntropyLoss().to(config['device'])
        return criterions
    
    if config['model_name'] == 'vsgnet':
        
        criterions = {}
        
        criterions['cr'] = nn.NLLLoss().to(config['device'])
        criterions['lr'] = nn.BCELoss().to(config['device'])
        criterions['mr'] = nn.BCELoss().to(config['device'])
        
        return criterions

    if config['model_name'] == 'ican': # or config['model_name'] == 'drg':
        
        criterions = {}
        
        criterions['cr'] = nn.NLLLoss().to(config['device'])
        criterions['lr'] = nn.BCELoss().to(config['device'])
        criterions['mr'] = nn.BCELoss().to(config['device'])
        
        return criterions

    if config['model_name'] == 'GPNN_icra' or config['model_name'] == 'graph_rcnn' or config['model_name'] == 'hgat':
        
        criterions = {}
        
        criterions['cr'] = nn.CrossEntropyLoss().to(config['device'])
        criterions['lr'] = nn.BCEWithLogitsLoss().to(config['device'])
        criterions['mr'] = nn.BCEWithLogitsLoss().to(config['device'])
        
        return criterions

    if config['model_name'] == 'mfurln' or config['model_name'] == 'imp':
        
        criterions = {}
        
        criterions['cr'] = nn.CrossEntropyLoss().to(config['device'])
        criterions['lr'] = nn.BCEWithLogitsLoss().to(config['device'])
        criterions['mr'] = nn.BCEWithLogitsLoss().to(config['device'])
        
        return criterions



    if config['model_name'] == 'drg':
        
        criterions = {}
        
        criterions['cr'] = nn.CrossEntropyLoss().to(config['device'])
        criterions['lr'] = nn.BCELoss().to(config['device'])
        criterions['mr'] = nn.BCELoss().to(config['device'])
        
        return criterions



    criterions = {}
    
    # criterions['cr'] = nn.CrossEntropyLoss().to(config['device'])
    # criterions['lr'] = nn.BCEWithLogitsLoss().to(config['device'])
    # criterions['mr'] = nn.BCEWithLogitsLoss().to(config['device'])
    
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
        

from model.nn_nets.graph_rcnn import graph_rcnn
from model.nn_nets.hgat import hgat
from model.nn_nets.mfurln import mfurln
from model.nn_nets.imp import imp

def get_model(config):

    if config['model_name'] == 'graph_rcnn':
        model = graph_rcnn(config).to(config['device'])
        return model

    if config['model_name'] == 'mfurln':
        model = mfurln(config).to(config['device'])
        return model

    if config['model_name'] == 'imp':
        model = imp(config).to(config['device'])
        return model


    if config['model_name'] == 'hgat':
        model = hgat(config).to(config['device'])
        return model

    if config['model_name'] == 'action_recog_test':
        model = action_net_cnn_stream(config).to(config['device'])
        return model

    if config['model_name'] == 'ican':

        model = iCAN(config).to(config['device'])
        return model

    if config['model_name'] == 'drg':

        model = DRG(config).to(config['device'])
        return model

    if config['model_name'] == 'vsgnet':

        model = vsgnet(config).to(config['device'])
        return model

    if config['model_name'] == 'ooi_net':

        model = ooi_net(config).to(config['device'])
        return model

    if config['model_name'] == 'GPNN':

        model = GPNN(config).to(config['device'])
        return model

    if config['model_name'] == 'GPNN_icra':

        model = GPNN(config).to(config['device'])
        return model

from dataloader.ican_dataset import ican_dataset

def get_dataset(config, split='train'):

    if config['dataset_description'] == 'action_dataset':
        
        return action_recog_dataset(config, split)
                    
    if config['dataset_description'] == 'ooi_dataset':

        return dataset(config, split)

    if config['dataset_description'] == 'vsgnet_dataset':

        return vsgnet_dataset(config, split)

    # if config['dataset_description'] == 'gpnn_dataset':

    #    # return gpnn_dataset(config, split)

    if config['dataset_description'] == 'gpnn_icra_dataset':

        return dataset(config, split)

    if config['dataset_description'] == 'ican_dataset':

        return ican_dataset(config, split)


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