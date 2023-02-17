from utils.utils import decollate_node_embeddings

from model.nn_modules.relation_classifier import relation_classifier
from model.nn_modules.gnn import GNN 

from utils.utils import aggregate

import torch.nn as nn
import torch.nn.functional as F
import torch


'''
How do we do action recognition?
Why do we do action recognition?
'''





























class action_net_single_stream(nn.Module):

    def __init__(self, config):
        '''
        Constructor for action_net_single_stream
        Args:
        ooi_net_config_file : Has the various configurations necessary for this network
                              This config file has information on the static and temporal feature
                              dimensions along with other things.
        '''
        super(action_net_single_stream, self).__init__()
        self.config = config

        # creating the gcn
        self.device = config['device']
        
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



class action_net_double_stream(nn.Module):

    def __init__(self, config):
        '''
        Constructor for action_net_double_stream
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
