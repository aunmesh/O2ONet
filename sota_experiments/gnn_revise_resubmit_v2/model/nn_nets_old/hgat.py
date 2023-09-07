import os

import torch
import torch.nn as nn
import torch.autograd

from utils.utils import collate_edge_indices, collate_node_features, decollate_node_embeddings

from model.nn_modules.gnn import GNN
from model.nn_modules.relation_classifier_2 import relation_classifier_2
from utils.utils import aggregate

class hgat(torch.nn.Module):

    def __init__(self, config):

        super(hgat, self).__init__()

        self.config = config.copy()      
        self.classifier_input_dimension = self.config['node_gnn_dimensions'][-1]
        
        self.node_gnn = GNN(self.config, 'node_')
        self.triplet_gnn = GNN(self.config, 'triplet_')
        
        # creating the cr classifier
        cr_dim = self.config['cr_dimensions']
        scr_dropout = self.config['cr_dropout']

        self.cr_cls = relation_classifier_2(cr_dim, scr_dropout,
                                            self.config['device'], 1)

        # creating the lr classifier
        lr_dim = self.config['lr_dimensions']
        lr_dropout = self.config['lr_dropout']

        self.lr_cls = relation_classifier_2(lr_dim, lr_dropout,
                                            self.config['device'], 1)

        # creating the mr classifier
        mr_dim = self.config['mr_dimensions']
        mr_dropout = self.config['mr_dropout']
        self.mr_cls = relation_classifier_2(
            mr_dim, mr_dropout, self.config['device'], 1)

        # Hyperparameters to process node embeddings for classification
        self.agg = self.config['aggregator']


    def make_linear_layer(self, dimensions):

        classifier_layers = []  # for storing all the transform layers

        for i in range(len(dimensions) - 1):
            curr_d = dimensions[i]
            next_d = dimensions[i+1]
            temp_fc_layer = nn.Linear(curr_d, next_d)
            classifier_layers.append(temp_fc_layer)
            
            if i < len(dimensions) - 2:
                classifier_layers.append(nn.ReLU())
                
        model = nn.Sequential(*classifier_layers).to(self.config['device'])

        return model


    def make_classifier_inputs(self, node_embeddings, triplet_embeddings, pairs):
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
            num_batches, num_pairs, 2*self.classifier_input_dimension, device=self.config['device'])

        for b in range(num_batches):

            for i in range(num_pairs):

                ind0, ind1 = pairs[b, i, 0], pairs[b, i, 1]

                emb0, emb1 = node_embeddings[b, ind0], node_embeddings[b, ind1]
                classifier_input[b, i, :self.classifier_input_dimension] = aggregate(emb0, emb1, self.agg)
                
                triplet_embedding = triplet_embeddings[b, i]
                classifier_input[b,i, self.classifier_input_dimension:] = triplet_embedding

        return classifier_input


    def forward(self, data_item):
        """
        get the edge index in the data_item
        Collate the node features and collate the edge indices. 
        Perform the graph attention based convolution.
        
        Construct the triplet graph. But between what relations. Use the NENN line adj mat?
        Doing convolution on the 
        
        data_item keys:
        ['metadata', 'geometric_feature', 'appearance_feature', 'i3d_feature',
        'motion_feature', 'num_obj', 'edge_index', 'num_edges', 'object_pairs',
        'scr', 'lr', 'mr', 'obj_mask', 'obj_pairs_mask', 'num_pairs', 'scr_metric', 
        'lr_metric', 'mr_metric', 'relative_feature', 'nenn_edge_index', 'nenn_num_edges', 
        'line_adj_mat', 'adj_mat']
        
        + ['concatenated_node_features', 'relative_spatial_feature']
        """
        
        batch_size = data_item['concatenated_node_features'].shape[0]
        
        data_item['edge_index'] = data_item['edge_index'].to(self.config['device'])
        data_item['num_edges'] = data_item['num_edges'].to(self.config['device'])
        data_item['num_obj'] = data_item['num_obj'].to(self.config['device'])
        
        # Perform the convolution on the node based graph
        collated_edge_index, edge_slicing = collate_edge_indices(
                                                                data_item['edge_index'],
                                                                data_item['num_edges'], 
                                                                data_item['num_obj'], 
                                                                self.config['device']
                                                                )
        
        collated_node_feature, node_slicing = collate_node_features(
                                                        data_item['concatenated_node_features'],
                                                        data_item['num_obj'], self.config['device']
                                                                    )

        # convolving the obj_features
        all_obj_embeddings = self.node_gnn(collated_node_feature, collated_edge_index)
        
        
        obj_embeddings = decollate_node_embeddings(all_obj_embeddings, node_slicing, 
                                                   self.config['device'])
        
        
        ## Perform the convolution on the triplet graph
        triplet_edge_index = torch.zeros((batch_size, 2, 250), dtype=torch.long, 
                                         device=self.config['device'])
        triplet_num_edges = torch.zeros((batch_size), device=self.config['device'])
        
        for b in range(batch_size):
            
            temp_adj_mat = data_item['line_adj_mat'][b]
            
            indices_start, indices_end = torch.where(temp_adj_mat)
            temp_num_edges = indices_start.shape[0]
            
            triplet_num_edges[b] = temp_num_edges
            
            triplet_edge_index[b,0,:temp_num_edges] = indices_start
            triplet_edge_index[b,1,:temp_num_edges] = indices_end
        

        collated_triplet_edge_index, triplet_edge_slicing = collate_edge_indices(triplet_edge_index,
                                                           triplet_num_edges,
                                                           data_item['nenn_num_edges'], 
                                                           self.config['device'])
        
        
        triplet_features = torch.zeros(( batch_size, 250, self.config['edge_feature_size']),
                                        device=self.config['device'])
        
        
        data_item['nenn_num_edges'] = data_item['nenn_num_edges'].to(self.config['device'])
        data_item['nenn_edge_index'] = data_item['nenn_edge_index'].to(self.config['device'])
        
        for b in range(batch_size):
            
            temp_num_edges = data_item['nenn_num_edges'][b]
            temp_indices_0 = data_item['nenn_edge_index'][b, 0, 0:temp_num_edges]
            temp_indices_1 = data_item['nenn_edge_index'][b, 1, 0:temp_num_edges]
            
            temp_features = data_item['interaction_feature'][b, temp_indices_0, temp_indices_1]
                                   
            triplet_features[b, :temp_num_edges] = temp_features
        
        collated_triplet_features, triplet_slicing = collate_node_features(triplet_features, 
                                                          data_item['nenn_num_edges'],
                                                          device=self.config['device']
                                                          )

        all_triplet_embeddings = self.triplet_gnn(collated_triplet_features, 
                                                  collated_triplet_edge_index)


        triplet_embeddings = decollate_node_embeddings(
                                                        all_triplet_embeddings,
                                                        triplet_slicing,
                                                        self.config['device'],
                                                        pad_len=250
                                                       )

        classifier_input = self.make_classifier_inputs(
                                                        obj_embeddings, 
                                                        triplet_embeddings,
                                                        data_item['object_pairs']
                                                       )

        predictions = {}
        predictions['combined'] = {}

        # Make the batch for features
        predictions['combined']['lr'] = self.lr_cls(classifier_input)
        predictions['combined']['cr'] = self.cr_cls(classifier_input)
        predictions['combined']['mr'] = self.mr_cls(classifier_input)

        return predictions