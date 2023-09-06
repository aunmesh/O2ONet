from model.nn_modules.mr_cls2 import mr_cls1
from utils.utils import decollate_node_embeddings
from utils.utils import config_loader
from model.nn_modules.scr_cls2 import scr_cls1
from model.nn_modules.lr_cls2 import lr_cls1
from model.nn_modules.nenn_test import NENN
import torch.nn as nn
import torch.nn.functional as F
import torch

import os
import sys
from utils.utils import aggregate



class nenn(nn.Module):
    '''
    ooi_net_1 stands for object-object interaction network 1.
    Uses:
        vanilla_gcn
        mr_cls_1
    '''

    def __init__(self, config):
        '''
        Constructor for ooi_net_1
        Args:
        ooi_net_config_file : Has the various configurations necessary for this network
                              This config file has information on the static and temporal feature
                              dimensions along with other things.
        '''
        super(nenn, self).__init__()
        
        self.config = config

        # creating the gcn
        self.device = config['device']
        self.nenn = NENN(config)

        self.agg = config['aggregator']

        # creating the scr classifier
        scr_dim = config['scr_dimensions']
        scr_dropout = config['scr_dropout']
        self.scr_cls = scr_cls1(scr_dim, scr_dropout, self.device)

        # creating the lr classifier
        lr_dim = config['lr_dimensions']
        lr_dropout = config['lr_dropout']
        self.lr_cls = lr_cls1(lr_dim, lr_dropout, self.device)

        # creating the mr classifier
        mr_dim = config['mr_dimensions']
        mr_dropout = config['mr_dropout']
        self.mr_cls = mr_cls1(mr_dim, mr_dropout, self.device)

        self.classifier_input_dimension = 2*config['nenn_node_out_dim_wn'] + 2*config['nenn_edge_out_dim_we']



    def make_classifier_inputs(self, node_embeddings, pairs, edge_embeddings):
        '''
        makes the classifier input from the node embeddings and pairs

        node_embeddings: Embeddings of the various nodes

        pairs: list of object pairs between which we have to do classification. 
               the object pairs are actually indices in the node_embeddings rows.

        pairs: A tensor of shape [b_size, MAX_PAIRS, 2]
               b_size is batch size, MAX_PAIRS is the maximum no. of pairs
        '''

        # Not implemented yet, checking whether the input
        # dimension of classifier matches the node embedding
        # Assume that an entire batch is coming

        num_batches = node_embeddings.shape[0]

        num_pairs = pairs.shape[1]   # Always equal to max pairs

        # classifier_input is the tensor which will be passed to the fully connected classifier
        # for feature classification
        classifier_input = torch.empty(
            num_batches, num_pairs, self.classifier_input_dimension, device=self.device)

        for b in range(num_batches):

            for i in range(num_pairs):

                ind0, ind1 = pairs[b, i, 0], pairs[b, i, 1]

                edge_embedding = edge_embeddings[b, ind0, ind1, :]
                
                emb0, emb1 = node_embeddings[b, ind0], node_embeddings[b, ind1]
                classifier_input[b, i] = torch.cat((aggregate(emb0, emb1, self.agg), 
                                                    edge_embedding), dim=0)

        return classifier_input


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
        
        node_features = data_item['concatenated_node_features'] 
        edge_index = data_item['nenn_edge_index'] 
        obj_pairs = data_item['object_pairs'] 
        edge_features = data_item['interaction_feature'] 
        num_objs = data_item['num_obj']
        num_edges = data_item['nenn_num_edges']
        line_adj_matrix = data_item['line_adj_mat']
        adj_matrix = data_item['adj_mat']
        
        
        b_size, num_unmasked_objects = node_features.shape[0], node_features.shape[1]

        nenn_embedding_size = self.config['nenn_edge_out_dim_we'] + self.config['nenn_node_out_dim_wn']


        all_node_embeddings = torch.zeros(b_size, num_unmasked_objects, nenn_embedding_size,
                                         dtype = node_features.dtype,
                                         device = self.config['device'])
        
        all_edge_embeddings = torch.zeros(b_size, num_unmasked_objects,num_unmasked_objects,
                                          nenn_embedding_size, dtype = node_features.dtype,
                                         device = self.config['device'])

        
         
        for b in range(b_size):
            
            temp_num_objs = num_objs[b]
            temp_num_edges = num_edges[b]
            
            temp_node_features = node_features[b][:temp_num_objs,:]
            temp_edge_index = edge_index[b][:,:temp_num_edges]
            temp_line_adj_matrix = line_adj_matrix[b][:temp_num_edges, :temp_num_edges]
            temp_adj_matrix = adj_matrix[b][:temp_num_objs, :temp_num_objs]
            temp_edge_features = edge_features[b]
            
            temp_node_embedding, temp_edge_embedding = self.nenn(temp_node_features, temp_edge_index,
                                                      temp_line_adj_matrix, temp_adj_matrix,
                                                      temp_edge_features)
            
            all_node_embeddings[b,:temp_num_objs,:] = temp_node_embedding
            all_edge_embeddings[b] = temp_edge_embedding


        classifier_input = self.make_classifier_inputs(
            all_node_embeddings, obj_pairs, all_edge_embeddings)

        predictions = {}
        predictions['combined'] = {}

        # Make the batch for features
        predictions['combined']['lr'] = self.lr_cls(classifier_input)
        predictions['combined']['cr'] = self.scr_cls(classifier_input)
        predictions['combined']['mr'] = self.mr_cls(classifier_input)

        return predictions