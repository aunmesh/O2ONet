from torch_geometric.nn import TransformerConv, GCNConv, GATConv

def get_gnn(config, in_dim, out_dim):
    
    if config['GNN'] == 'TransformerConv':
        return TransformerConv(in_dim, out_dim)
    
    if config['GNN'] == 'GCN':
        return GCNConv(in_dim, out_dim)
    
    if config['GNN'] == 'GATConv':
        return GATConv(in_dim, out_dim)

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
                        default="/workspace/work/O2ONet/sota_experiments/gnn/configs/training.yaml",
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



def process_data_for_fpass(data_item, config):
   
   # obj_features, obj_pairs, slicing dictionary
   
   
    # Pre-process the feature tensors
    
    central_frame_index = int(config['gif_size']/2)

    # i3d feature - perform average pool
    # Incoming shape - [batch_size, num_frames, num_obj, f_size]
    data_item['i3d_feature'] = torch.mean(data_item['i3d_feature_map'], 1)

    # 2d cnn feature - nothing needs to be done

    # geometric_feature - select the central frame
    # shape: [batch_size, num_obj, num_frames, f_size]
    
    # data_item['central_frame_geometric_feature'] = data_item['geometric_feature'][:, :, central_frame_index, :]
    # data_item['central_frame_motion_feature'] = data_item['motion_feature'][:, :, central_frame_index, :]

    data_item['central_frame_geometric_feature'] = data_item['geometric_feature'].flatten(2)
    data_item['central_frame_motion_feature'] = data_item['motion_feature'][:, :, central_frame_index, :].flatten(2)


    # Padded features
    obj_features = [ data_item[f] for f in config['features_list'] ]
    obj_features = torch.cat(obj_features, 2)


    # Padded Edge Index

    collated_obj_features, node_slicing = collate_node_features(obj_features, 
                                                    data_item['num_obj'], config['device']
                                                        )

    collated_edge_index, edge_slicing = collate_edge_indices(data_item['edge_index'], 
                            data_item['num_edges'], data_item['num_obj'], config['device']
                                                        )

    data_item['edge_index'] = data_item['edge_index'].to(config['device'])
    data_item['num_edges'] = data_item['num_edges'].to(config['device'])
    data_item['num_obj'] = data_item['num_obj'].to(config['device'])

    data_item['collated_obj_features'] = collated_obj_features.to(config['device'])
    data_item['collated_obj_features'] = data_item['collated_obj_features'].type(torch.double)
    
    data_item['collated_edge_index'] = collated_edge_index.to(config['device'])

    # Create the slicing dictionary in train.py from incoming 
    # features and incoming edge indices.

    data_item['slicing'] = {}

    data_item['slicing']['node'] = node_slicing
    data_item['slicing']['edge'] = edge_slicing

    data_item['object_pairs'] = data_item['object_pairs'].type(torch.long)
    
    data_item['lr'] = data_item['lr'].to(config['device'])
    data_item['mr'] = data_item['mr'].to(config['device'])
    data_item['cr'] = data_item['cr'].to(config['device'])
    
    return data_item


def process_data_for_metrics(d_item):
    # function for further processing of the data item d_item
    # # obtained from the data_loader 
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




import torch
from zmq import device

def collate_node_features(obj_features, num_objects, device):
    
    total_objects = torch.sum(num_objects)
    dim_features = obj_features.shape[-1] #[b_size, num_obj, feature_dimension]

    res = torch.zeros((total_objects, dim_features), dtype = obj_features.dtype, device=device)
    
    b_size = int(obj_features.shape[0])
    curr_index = 0
    node_slices = torch.zeros((total_objects), device=device)

    for b in range(b_size):
        
        curr_obj = int(num_objects[b])
        res[curr_index: curr_index + curr_obj,:] = obj_features[b,:curr_obj,:]
        
        node_slices[curr_index:curr_index + curr_obj] = b

        curr_index+=curr_obj
    
    return res, node_slices

def collate_edge_indices(edge_index, num_edges, num_objects, device):

    total_edges = torch.sum(num_edges)
    res = torch.zeros((2, total_edges), dtype = edge_index.dtype, device=device)

    b_size = int(edge_index.shape[0])

    curr_index = 0
    edge_slices = torch.zeros((total_edges), device=device)
    lower = 0
    upper = 0
    for b in range(b_size):

        curr_edge = int(num_edges[b])

        temp_num_objects = torch.sum(num_objects[lower:upper])

        res[ : , curr_index: curr_index+curr_edge] = temp_num_objects
        res[ : , curr_index: curr_index+curr_edge] += edge_index[b, :, : curr_edge]

        edge_slices[curr_index: curr_index+curr_edge] = b
        upper+=1        
        curr_index+=curr_edge

    return res, edge_slices



def decollate_node_embeddings(all_node_embeddings, node_slicing, device, pad_len=8):

    dim_embedding = all_node_embeddings.shape[-1]
    b_size = int(node_slicing[-1]+1)
    
    result = torch.zeros((b_size, pad_len, dim_embedding), dtype=all_node_embeddings.dtype, device=device)
    
    for i in range(b_size):
        curr_obj = i
        
        indices = torch.where(node_slicing == curr_obj)
        num_obj = indices[0].shape[0]

        temp_embeddings = all_node_embeddings[indices]
        result[i][:num_obj] = temp_embeddings

    return result


import torch

def concatenate(t1, t2):
    return torch.cat((t1, t2))

def mean(t1, t2):
    return t1.add(t2)/2.0

def aggregate(t1, t2, name):
    
    assert name == "mean" or name == "concat", "aggregator name can only be mean or concat"

    if name=='mean':
        return mean(t1, t2)
    
    if name=='concat':
        return concatenate(t1, t2)