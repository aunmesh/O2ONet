from os import device_encoding
import yaml
import pickle
import torch

def config_loader(config_file_location):

    f = open(config_file_location)
    #config = yaml.load(f, Loader=yaml.FullLoader)
    config = yaml.safe_load(f)
    f.close()
    config['device'] = torch.device(config['device'])

    return config

import os
from glob import glob
import numpy as np

def get_sorted_filelist(folder_path):

    if folder_path[-1] == '/':
        folder_path = folder_path + '*'
    else:
        folder_path = folder_path +  '/*'

    file_list = glob(folder_path)
    file_list_sorted = list(np.sort(file_list))

    return file_list_sorted



def read_data(file_location):

    with open(file_location, 'rb') as handle:
        data = pickle.load(handle)
        handle.close()

    return data


def args_to_configs(config, args):
    
    args = args.__dict__
    arg_keys = args.keys()
    config_keys = config.keys()
    
    if 'gcn_dropout' in config_keys:
        config['gcn_dropout'] = args['dropout']
        config['lr_dropout'] = args['dropout']
        config['mr_dropout'] = args['dropout']
        config['scr_dropout'] = args['dropout']

        #---------------------#
            
        gcn_hidden_dim = int(args['gcn_hidden_dim'])
        gcn_hidden_layers = int(args['gcn_hidden_layers'])
        gcn_input_dim = int(config['gcn_dimensions'][0])
        gcn_output_dim = int(args['gcn_output_dim'])

        config['gcn_dimensions'] = [gcn_input_dim] + [gcn_hidden_dim]*gcn_hidden_layers + [gcn_output_dim]
        
        #---------------------#
        
        lr_input_dim = int(args['gcn_output_dim'])
        lr_hidden_dim = int(args['lr_hidden_dim'])
        lr_hidden_layers = int(args['lr_hidden_layers'])
        lr_output_dim = 5

        config['lr_dimensions'] = [lr_input_dim] + [lr_hidden_dim]*lr_hidden_layers + [lr_output_dim]

        #---------------------#

        mr_input_dim = int(args['gcn_output_dim'])
        mr_hidden_dim = int(args['mr_hidden_dim'])
        mr_hidden_layers = int(args['mr_hidden_layers'])
        mr_output_dim = 14

        config['mr_dimensions'] = [mr_input_dim] + [mr_hidden_dim]*mr_hidden_layers + [mr_output_dim]

        #---------------------#

        scr_input_dim = int(args['gcn_output_dim'])
        scr_hidden_dim = int(args['scr_hidden_dim'])
        scr_hidden_layers = int(args['scr_hidden_layers'])
        scr_output_dim = 3

        config['scr_dimensions'] = [scr_input_dim] + [scr_hidden_dim]*scr_hidden_layers + [scr_output_dim]

    for k in config_keys:
        if k in arg_keys:
            config[k] = args[k]
    
    return config

import argparse

def get_parser():

    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--train',
                        type=bool, 
                        default=True, 
                        help='bool for training'
                        )

    parser.add_argument('--config', 
                        default="/workspace/work/CVPR22/action_understanding/config/training.yaml",
                        help='config file location')

    parser.add_argument('--resume',
                        type=bool, 
                        default=False, 
                        help='bool for training'
                        )

    parser.add_argument('--run_id',
                        type=str, 
                        default='', 
                        help='bool for training'
                        )

    return parser



def pool_edge_features(edge_feature_mat, edge_index):
    
    edge_feature_dim = edge_feature_mat.shape[-1]
    num_edges = edge_index.shape[-1]
    result = torch.zeros(edge_feature_dim)

    for k in range(num_edges):
        temp_node0, temp_node1 = int(edge_index[0, k]), int(edge_index[1, k])
        temp_feature = edge_feature_mat[temp_node0, temp_node1]
        result = result + temp_feature
    
    result = result/num_edges
    
    return result



def process_data_for_fpass(d_item, config):
    # function for further processing of the data item d_item
    # # obtained from the data_loader 

    '''
    #torch.Size([500, 11, 15, 5]) #torch.Size([500, 15, 2048])
    '''
    # for k in d_item.keys():
    #     try:
    #         print(k, d_item[k].shape)
    #     except:
    #         print(k)

    obj_features_temporal = []
    for f in config['ooi_node_temporal_features_list']:
        
        temp_shape = d_item[f].shape
        temp_tensor = d_item[f].transpose(2,3).reshape(temp_shape[0],temp_shape[1], temp_shape[3], -1)
        obj_features_temporal.append(temp_tensor)


    obj_features_nontemporal = [ d_item[f] for f in config['ooi_node_nontemporal_features_list'] ]    

    obj_features = torch.cat(obj_features_temporal + obj_features_nontemporal, 3)

    d_item['clip_node_features'] = obj_features.to(config['device'])
    d_item['clip_edge_index'] = d_item['clip_edge_index'].to(config['device'])

    d_item['feature_mat'] = d_item['video_i3d_feature'].to(config['device'])
    d_item['feature_mat'] = d_item['feature_mat'].transpose(1,2)

    d_item['label_vector'] = d_item['clip_label_vector'].long().to(config['device'])
    
    #d_item['num_frames'] = d_item['num_clips'].to(config['device'])
    b_size = d_item['feature_mat'].shape[0]
    
    d_item['frames_mask'] = torch.zeros(b_size, 500).to(config['device'])

    for b in range(b_size):
        temp_num_frames = d_item['num_clips'][b]
        d_item['frames_mask'][b, :temp_num_frames] = 1
    
    d_item['num_clips'] = d_item['num_clips'].int().to(config['device'])

    d_item['frames_mask'][b, :temp_num_frames] = 1

    
    if ('relative_feature' in config.keys()):
        temp_mat = d_item['clip_re_feature']
        
        d_item['relative_feature_pooled'] = torch.zeros(b_size, 77, 500).to(config['device'])

        for b in range(b_size):
            temp_num_clips = d_item['num_clips'][b]
            temp_batch_feature_mat = d_item['clip_re_feature'][b,:temp_num_clips] #num_clips,11,15,15,7
            for i in range(temp_num_clips):
                temp_mat = temp_batch_feature_mat[i]
                temp_mat = temp_mat.swapaxes(0,1)
                temp_mat = temp_mat.swapaxes(1,2)
                temp_mat = temp_mat.contiguous()
                temp_mat = temp_mat.view(15,15,-1)
                edge_index = d_item['clip_edge_index'][b,i]
                result = pool_edge_features(temp_mat, edge_index)
                d_item['relative_feature_pooled'][b,:,i] = result
    
    if 'nenn' in config['model_name']:
        d_item['clip_nenn_edge_index'] = d_item['nenn_edge_index'].to(config['device'])
        d_item['clip_nenn_num_edges'] = d_item['nenn_num_edges'].to(config['device'])
        d_item['clip_re_feature'] = d_item['clip_re_feature'].to(config['device'])

    return d_item


def process_data_for_metrics(d_item):
    # function for further processing of the data item d_item
    # # obtained from the data_loader 
    raise NotImplementedError


def loss_aggregator(loss_dict, dset_size):
    raise NotImplementedError




class loss_epoch_aggregator:

    def __init__(self, stage='train'):
        self.loss_aggregated = None
        self.count=0.0
        self.stage = stage + '_'
    
    def add(self, loss_dict):
        self.count+=1

        if self.loss_aggregated == None:

            self.loss_aggregated = {}
            incoming_keys = loss_dict.keys()

            for k in incoming_keys:
                self.loss_aggregated[self.stage + k] = loss_dict[k]

        else:
            curr_keys = self.loss_aggregated.keys()
            incoming_keys = loss_dict.keys()
            
            for k in incoming_keys:
                
                if k in curr_keys:
                    self.loss_aggregated[self.stage + k] += loss_dict[k]
                
                else:
                    self.loss_aggregated[self.stage + k] = loss_dict[k]
    
    def average(self):
        
        for k in self.loss_aggregated.keys():
            self.loss_aggregated[k] = self.loss_aggregated[k]/(self.count)
        
        return self.loss_aggregated
    
    def reset(self):
        self.loss_aggregated = None
        self.stage = ''