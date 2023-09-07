from torch_geometric.nn import TransformerConv, GCNConv, GATConv, GATv2Conv
from executor.loss import *

def get_loss(output, target, criterions, config=None):
        
    if config['model_name'] == 'graph_rcnn':
        return masked_loss_graph_rcnn(output, target, criterions)

    if config['model_name'] == 'SQUAT':
        return masked_loss_squat(output, target, criterions)

        
    return masked_loss_gpnn(output, target, criterions)        

def get_gnn(config, in_dim, out_dim):
    
    if config['GNN'] == 'TransformerConv':
        return TransformerConv(in_dim, out_dim)
    
    if config['GNN'] == 'GCN':
        return GCNConv(in_dim, out_dim)
    
    if config['GNN'] == 'GATConv':
        return GATConv(in_dim, out_dim)

    if config['GNN'] == 'GATv2Conv':
        return GATv2Conv(in_dim, out_dim)


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
    
    if config['model_name'] == 'GPNN_icra':
        num_frames = int(config['num_frames'])
        config['edge_feature_size'] = 7 * num_frames

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
    
    parser.add_argument('--stratified',
                        type=int, 
                        default=0, 
                        help='stratified split number'
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

def process_data_for_fpass_double(data_item, config):
      
    data_item['num_relation'] = data_item['num_relation'].to(config['device'])
    data_item['num_obj'] = data_item['num_obj'].to(config['device'])

    data_item['object_pairs'] = data_item['object_pairs'].type(torch.long)
    
    data_item['lr'] = data_item['lr'].to(config['device'])
    data_item['mr'] = data_item['mr'].to(config['device'])
    data_item['cr'] = data_item['cr'].to(config['device'])

    # Padded features
    obj_features = []
    
    for f in config['features_list']:
        
        if f in config['custom_filter_dict'].keys():
            
            # increased by 1 to take care of the batching dimension
            frame_dim = config['custom_filter_dict'][f]['frame_dim'] + 1
            
            frame_index = config['custom_filter_dict'][f]['frame_index']
            frame_index = torch.tensor([frame_index])
            
            temp_feat = data_item[f].index_select( 
                                                    dim = frame_dim,
                                                    index = frame_index
                                                ).squeeze().to(config['device']).double()
            
            obj_features.append(temp_feat)
        
        else:
            temp_feat = data_item[f].to(config['device']).double()
            obj_features.append(temp_feat)
    
    obj_features = torch.cat(obj_features, 2)

    data_item['concatenated_node_features'] = obj_features.to(config['device']).double()

    interaction_centric_features = []
    
    for f in config['relative_features_list']:
        temp_feat = data_item[f].flatten(3).to(config['device']).double()
        interaction_centric_features.append(temp_feat)

    data_item['interaction_feature'] = torch.cat(interaction_centric_features, 3)
    
    return data_item

def convert_tensor_values_to_float(input_dict):
    """
    Converts all values in the dictionary that are tensors to float dtype,
    except for tensors of dtype torch.long.
    """
    # Initialize an empty dictionary to store the results
    output_dict = {}
    
    for key, value in input_dict.items():
        # Check if the value is a tensor and not of type torch.long
        if isinstance(value, torch.Tensor) and value.dtype != torch.long:
            # Convert the tensor value to float and update in the new dictionary
            output_dict[key] = value.float()
        else:
            # If the value is not a tensor or is of type torch.long, 
            # directly update in the new dictionary
            output_dict[key] = value
    
    return output_dict


def process_data_for_fpass(data_item, config):
      
    data_item['num_relation'] = data_item['num_relation'].to(config['device'])
    data_item['num_obj'] = data_item['num_obj'].to(config['device'])

    data_item['object_pairs'] = data_item['object_pairs'].type(torch.long)
    
    data_item['lr'] = data_item['lr'].to(config['device'])
    data_item['mr'] = data_item['mr'].to(config['device'])
    data_item['cr'] = data_item['cr'].to(config['device'])

    # Padded features
    obj_features = []
    
    for f in config['features_list']:
        
        if f in config['custom_filter_dict'].keys():
            
            # increased by 1 to take care of the batching dimension
            frame_dim = config['custom_filter_dict'][f]['frame_dim'] + 1
            
            frame_index = config['custom_filter_dict'][f]['frame_index']
            frame_index = torch.tensor([frame_index])
            
            temp_feat = data_item[f].index_select( 
                                                    dim = frame_dim,
                                                    index = frame_index
                                                ).squeeze().to(config['device'])
            
            obj_features.append(temp_feat)
        
        else:
            temp_feat = data_item[f].to(config['device'])
            obj_features.append(temp_feat)
    
    obj_features = torch.cat(obj_features, 2)

    data_item['concatenated_node_features'] = obj_features.to(config['device'])

    interaction_centric_features = []
    
    for f in config['relative_features_list']:
        temp_feat = data_item[f].flatten(3).to(config['device'])
        interaction_centric_features.append(temp_feat)

    data_item['interaction_feature'] = torch.cat(interaction_centric_features, 3)
    
    data_item = convert_tensor_values_to_float(data_item)

    return data_item


import torch





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
            curr_keys = list(self.loss_aggregated.keys())
            incoming_keys = list(loss_dict.keys())
            
            for k in incoming_keys:
                check_string = self.stage + k
                
                if check_string in curr_keys:
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

    try:
        total_edges = torch.sum(num_edges)
        res = torch.zeros((2, total_edges), dtype = torch.long, device=device)
    except:
        total_edges = int(torch.sum(num_edges).item())
        res = torch.zeros((2, total_edges), dtype = torch.long, device=device)
        
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