import os

import torch
import torch.nn as nn
import torch.autograd

from utils.utils import collate_edge_indices, collate_node_features, decollate_node_embeddings

from model.nn_modules.gnn import GNN
from model.nn_modules.relation_classifier_2 import relation_classifier_2
from utils.utils import aggregate
import torch.nn.functional as F

class imp(torch.nn.Module):

    def __init__(self, config):

        super(imp, self).__init__()

        self.config = config.copy()      
        
        
        hidden_size = 128
        self.classifier_input_dimension = hidden_size
        self.node_transform = self.make_linear_layer([config['node_feature_size'], 512, hidden_size])
        self.edge_transform = self.make_linear_layer([config['edge_feature_size'], hidden_size])
        
        self.node_gru = torch.nn.GRU(hidden_size,hidden_size, batch_first=True)
        self.edge_gru = torch.nn.GRU(hidden_size,hidden_size, batch_first=False)
        
        self.node_attention_net = self.make_linear_layer([2*hidden_size,1])
        self.edge_attention_net = self.make_linear_layer([2*hidden_size,1])
        
        # creating the cr classifier
        cr_dim = self.config['cr_dimensions']
        cr_dim[0] = 2*hidden_size
        scr_dropout = self.config['cr_dropout']

        self.cr_cls = relation_classifier_2(cr_dim, scr_dropout,
                                            self.config['device'], 1)
        self.cr_softmax = torch.nn.Softmax(dim=-1)

        # creating the lr classifier
        lr_dim = self.config['lr_dimensions']
        lr_dim[0] = 2*hidden_size
        lr_dropout = self.config['lr_dropout']
        self.lr_cls = relation_classifier_2(
            lr_dim, lr_dropout, self.config['device'], 1)

        # creating the mr classifier
        mr_dim = self.config['mr_dimensions']
        mr_dim[0] = 2*hidden_size
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


    def node_message_pooling(self, node_gru_hidden_states, edge_gru_hidden_states, data_item):
        
        # Perform attention based node message pooling
        batch_size = data_item['concatenated_node_features'].shape[0]
        num_nodes = data_item['concatenated_node_features'].shape[1]
        
        node_gru_hidden_states = node_gru_hidden_states.reshape([batch_size, num_nodes, -1])
        edge_gru_hidden_states = edge_gru_hidden_states.reshape([batch_size, num_nodes, num_nodes, -1])
        
        node_gru_dimension = node_gru_hidden_states.shape[-1]
        edge_gru_dimension = edge_gru_hidden_states.shape[-1]
        
        node_gru_hidden_states_augmented = node_gru_hidden_states.unsqueeze(-2).expand((batch_size, num_nodes,
                                                                                        num_nodes, node_gru_dimension))
        
        augmented_matrix = torch.cat([node_gru_hidden_states_augmented, edge_gru_hidden_states], dim=-1)
        attention_logits = self.node_attention_net(augmented_matrix).squeeze()
        attention_weights = F.softmax(attention_logits, dim=-1)
        
        messages = torch.einsum('bijk,bij -> bik', edge_gru_hidden_states, attention_weights)
        return messages
                   
    def edge_message_pooling(self, node_gru_hidden_states, edge_gru_hidden_states, data_item):

        # Perform attention based edge message pooling
        batch_size = data_item['concatenated_node_features'].shape[0]
        num_nodes = data_item['concatenated_node_features'].shape[1]
        
        node_gru_hidden_states = node_gru_hidden_states.reshape([batch_size, num_nodes, -1])
        edge_gru_hidden_states = edge_gru_hidden_states.reshape([batch_size, num_nodes, num_nodes, -1])
        
        node_gru_dimension = node_gru_hidden_states.shape[-1]
        edge_gru_dimension = edge_gru_hidden_states.shape[-1]
        
        node_gru_hidden_states_augmented_column = node_gru_hidden_states.unsqueeze(-2).expand((batch_size, num_nodes,
                                                                                        num_nodes, node_gru_dimension))

        node_gru_hidden_states_augmented_row = node_gru_hidden_states.unsqueeze(-3).expand((batch_size, num_nodes,
                                                                                        num_nodes, node_gru_dimension))
        
        augmented_matrix_row = torch.cat([node_gru_hidden_states_augmented_row, edge_gru_hidden_states],
                                         dim=-1).unsqueeze(-2)
        
        augmented_matrix_col = torch.cat([node_gru_hidden_states_augmented_column, edge_gru_hidden_states],
                                         dim=-1).unsqueeze(-2)
        
        node_gru_augmented = torch.cat([node_gru_hidden_states_augmented_row.unsqueeze(-1),
                                        node_gru_hidden_states_augmented_column.unsqueeze(-1)], dim=-1)
        
        augmented_matrix = torch.cat([augmented_matrix_row, augmented_matrix_col], dim=-2)
        attention_logits = self.edge_attention_net(augmented_matrix).squeeze() # shape - b,i,j,2
        attention_weights = F.softmax(attention_logits, dim=-1)
        message = torch.einsum('bijl, bijkl -> bijk', attention_weights, node_gru_augmented)
        return message


    def make_classifier_inputs(self, node_embeddings, edge_embeddings, pairs, 
                               num_rels):
        '''
        makes the classifier input from the node embeddings and pairs

        node_embeddings: Embeddings of the various nodes

        pairs: list of object pairs between which we have to do classification. 
               the object pairs are actually indices in the node_embeddings rows.

        pairs: A tensor of shape [b_size, MAX_PAIRS, 2]
               b_size is batch size, MAX_PAIRS is the maximum no. of pairs
        '''
        num_batches = node_embeddings.shape[0]
        num_pairs = pairs.shape[1]   # Always equal to max pairs

        classifier_input = torch.zeros(
                                        num_batches, num_pairs, 
                                        2*self.classifier_input_dimension, 
                                        device=self.config['device']
                                       )
        
        half_dimension = self.classifier_input_dimension

        for b in range(num_batches):
           
            temp_num_pairs = num_rels[b]
            for i in range(num_pairs):

                ind0, ind1 = pairs[b, i, 0], pairs[b, i, 1]
                emb0, emb1 = node_embeddings[b, ind0], node_embeddings[b, ind1]
                
                classifier_input[b, i, :half_dimension] = aggregate(emb0, emb1, self.agg)
                
                embed1 = edge_embeddings[b, ind0, ind1, :]
                embed2 = edge_embeddings[b, ind1, ind0, :]
                
                
                classifier_input[b, i, half_dimension:] = aggregate(embed1, embed2, self.agg)

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
        
        data_item['num_obj'] = data_item['num_obj'].to(self.config['device'])

        '''
        Algorithm:
        1. get the node features of the entire batch
        2. get the edge features of the entire batch
        3. pass them to gru with sequence length set to 1.
        4. Perform message pooling
        '''
        concatenated_node_features = data_item['concatenated_node_features']
        edge_features = data_item['interaction_feature']
        
        node_features = self.node_transform(concatenated_node_features)
        edge_features = self.edge_transform(edge_features)
        
        batch_size = concatenated_node_features.shape[0]
        num_nodes = concatenated_node_features.shape[1]
        
        num_time_steps = 2
        
        for t in range(num_time_steps):
            node_features = node_features.reshape([batch_size*num_nodes, -1])
            node_features = node_features.unsqueeze(1)
            
            edge_features = edge_features.reshape([batch_size*num_nodes*num_nodes, -1])
            edge_features = edge_features.unsqueeze(1)
            
            node_gru_hidden_states = self.node_gru(node_features)[0]
            edge_gru_hidden_states = self.edge_gru(edge_features)[0]
            
            node_gru_hidden_states = self.node_message_pooling(node_gru_hidden_states, 
                                                               edge_gru_hidden_states,
                                                               data_item)
            
            edge_gru_hidden_states = self.edge_message_pooling(node_gru_hidden_states,
                                                               edge_gru_hidden_states,
                                                               data_item)
        
        classifier_input = self.make_classifier_inputs(node_gru_hidden_states, edge_gru_hidden_states,
                                                       data_item['object_pairs'], data_item['num_relation'])
            
        
        # print("DEBUG edge_features shape", edge_features.shape)
        
        predictions = {}
        predictions['combined'] = {}

        # Make the batch for features
        predictions['combined']['lr'] = self.lr_cls(classifier_input)
        predictions['combined']['cr'] = self.cr_cls(classifier_input)
        predictions['combined']['mr'] = self.mr_cls(classifier_input)

        return predictions
