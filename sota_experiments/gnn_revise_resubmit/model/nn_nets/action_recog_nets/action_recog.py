from utils.utils import decollate_node_embeddings

from model.nn_modules.relation_classifier import relation_classifier
from model.nn_modules.gnn import GNN 

from utils.utils import aggregate, get_gnn

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch_geometric.nn.pool import SAGPooling

'''
How do we do action recognition?
Why do we do action recognition?
'''

class action_net_cnn_stream(nn.Module):

    def __init__(self, config):
        '''
        Constructor for action_net_single_stream
        Args:
        ooi_net_config_file : Has the various configurations necessary for this network
                              This config file has information on the static and temporal feature
                              dimensions along with other things.
        '''
        super(action_net_cnn_stream, self).__init__()
        self.config = config.copy()

        # creating the gcn
        self.device = config['device']
        
        self.hidden_layer = nn.Linear(2048, 128)
        self.classification_layer = nn.Linear(128, 11)

        
    def forward(self, data_item):
        '''
        Input:
            obj_features: These are actually node features
            adj_mat: The adjacency matrices
            obj_pairs: The pairs between which we have to predict relation

        Output:
            results: a dictionary with keys for mr,lr and scr. 
                    The keys point to results of mr, lr and scr classification.
        '''
        i3d_fmap = data_item['i3d_fmap']
        # i3d_fmap_mean = torch.mean(i3d_fmap) #, dim=(2,3,4))
        
        x = F.relu(self.hidden_layer(i3d_fmap))
        x = self.classification_layer(x)

        pred = {}
        pred['action_index_logit'] = x

        return pred

class action_net_gnn_stream(nn.Module):

    def __init__(self, config):
        '''
        Constructor for action_net_single_stream
        Args:
        ooi_net_config_file : Has the various configurations necessary for this network
                              This config file has information on the static and temporal feature
                              dimensions along with other things.
        '''
        super(action_net_gnn_stream, self).__init__()
        self.config = config.copy()

        ## Get the list of node features
        self.feature_list = self.config['node_features']


        # creating the gcn
        self.device = config['device']
        
        self.gnn_layers = nn.ModuleList()
        
        for i in range(self.num_gnn_blocks):
            
            temp_gnn_layer = get_gnn(self.config, 
                                     self.config['gnn_in_dim'],
                                     self.config['gnn_out_dim']
                                    ).to(self.config['device'])
            
            self.gnn_layers.append(temp_gnn_layer)
        
        self.pooling_layer = SAGPooling(self.config['gnn_out_dim'])
        
        # A tensor which is used in the forward pass below.
        # This tensor is mostly fixed, so we are defining 
        # it as an instance variable. 
        self.batch_tensors = {}
        
        self.classification_layer = nn.Linear(self.config['gnn_out_dim'], 11)
        
    def forward(self, data_item):
        '''
        Input:
            obj_features: These are actually node features
            adj_mat: The adjacency matrices
            obj_pairs: The pairs between which we have to predict relation

        Output:
            results: a dictionary with keys for mr,lr and scr. 
                    The keys point to results of mr, lr and scr classification.
        '''
        # Prepare the data for graph convolution
        
        ## Concatenate the node features
        obj_features = [ data_item[f] for f in self.feature_list ]
        obj_features = torch.cat(obj_features, 2).to(self.config['device'])

        ## Collate the node features
        node_feats_flat = obj_features.flatten(0, 1)

        # We assume that the collated edge index is given to us
        collated_edge_index = data_item['collated_edge_index']

        
        ## send to gnn to get the node embeddings
        for i in range(len(self.gnn_layers)):       
            node_feats_flat = self.gnn_layers[i](node_feats_flat, collated_edge_index)
            node_feats_flat = F.relu(node_feats_flat)
        
        ## Perform the graph pooling
        
        ### Get the batch tensor which has the batch information for every node
        batch_size, max_num_obj = obj_features.shape[0], obj_features.shape[1]
        key_str = str(batch_size) + '_' + str(max_num_obj)
        if key_str in self.batch_tensors.keys():
            batch_tensor = self.batch_tensors[key_str]
        else:
            batch_tensor = []
            for b in range(batch_size):
                for m in range(max_num_obj):
                    batch_tensor.append(batch_size)
            batch_tensor = torch.tensor(batch_tensor)
            key_str = str(batch_size) + '_' + str(max_num_obj)
            self.batch_tensors[key_str] = batch_tensor

        graph_embeddings = self.pooling_layer(
                                    node_feats_flat, 
                                    edge_index=collated_edge_index,
                                    batch=batch_tensor
                                )
        graph_embeddings = F.relu(graph_embeddings)
        classification_logits = self.classification_layer(graph_embeddings)
        
        pred = {}
        pred['action_index_logit'] = classification_logits

        return pred        

class action_net_double_stream(nn.Module):

    def __init__(self, config):
        '''
        Constructor for action_net_double_stream. This network has one stream which processes the frame level i3d information.
        The second stream processes the interaction information using a graph neural network.
        Args:
        ooi_net_config_file : Has the various configurations necessary for this network
                              This config file has information on the static and temporal feature
                              dimensions along with other things.
        '''
        super(action_net_double_stream, self).__init__()
        self.config = config

        # creating the gcn
        self.device = config['device']

        self.feat1_transform = None
        self.feat2_transform = None
                
        self.hidden_layer = None
        self.classification_layer = None


    def forward(self, data_item):
        '''
        Input:
            obj_features: These are actually node features
            adj_mat: The adjacency matrices
            obj_pairs: The pairs between which we have to predict relation

        Output:
            results: a dictionary with keys for mr,lr and scr. 
                    The keys point to results of mr, lr and scr classification.
        '''



class action_net_triple_stream(nn.Module):

    def __init__(self, config):
        '''
        Constructor for ooi_net_1
        Args:
        ooi_net_config_file : Has the various configurations necessary for this network
                              This config file has information on the static and temporal feature
                              dimensions along with other things.
        '''
        super(action_net_triple_stream, self).__init__()
        self.config = config

        # creating the gcn
        self.device = config['device']


        self.feat1_transform = None
        self.feat2_transform = None
        self.feat3_transform = None                
        self.hidden_layer = None
        self.classification_layer = None


    def forward(self, data_item):
        '''
        Input:
            obj_features: These are actually node features
            adj_mat: The adjacency matrices
            obj_pairs: The pairs between which we have to predict relation

        Output:
            results: a dictionary with keys for mr,lr and scr. 
                    The keys point to results of mr, lr and scr classification.
        '''
        
        



class action_net_multi_stream(nn.Module):
    
    def __init__(self, config):
        '''
        Constructor for ooi_net_1
        Args:
        ooi_net_config_file : Has the various configurations necessary for this network
                              This config file has information on the static and temporal feature
                              dimensions along with other things.
        '''
        super(action_net_multi_stream, self).__init__()
        self.config = config

        # creating the gcn
        self.device = config['device']


        self.feature_transforms = []
                      
        self.hidden_layer = None
        self.classification_layer = None


    def forward(self, data_item):
        '''
        Input:
            obj_features: These are actually node features
            adj_mat: The adjacency matrices
            obj_pairs: The pairs between which we have to predict relation

        Output:
            results: a dictionary with keys for mr,lr and scr. 
                    The keys point to results of mr, lr and scr classification.
        '''
