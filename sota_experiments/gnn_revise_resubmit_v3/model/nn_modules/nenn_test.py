import sys
sys.path.append('/workspace/work/CVPR22/ooi_classification/nenn')

from numpy import require
import torch.nn as nn
import torch.nn.functional as F
import torch
from typepy import Bool
from utils.nenn_utils import *


class NENN(nn.Module):

    def __init__(self, config, stage=0):
        '''
        creates a graph neural network
        dimensions are a list for [input dimension, hidden....hidden, output dimension]
        drouput is the dropout probability. Same for all layers
        '''

        super(NENN, self).__init__()
        
        self.config = config
        self.stage = stage

        if stage == 0:
            self.w_n = nn.Linear(config['nenn_node_in_dim_wn'], config['nenn_node_out_dim_wn']).to(self.config['device'])
            self.w_e = nn.Linear(config['nenn_edge_in_dim_we'], config['nenn_edge_out_dim_we']).to(self.config['device'])
            
            self.a_t_n = nn.Linear(2*config['nenn_node_out_dim_wn'], 1).to(self.config['device']).to(self.config['device'])
            self.a_t_e = nn.Linear(config['nenn_edge_out_dim_we']+config['nenn_node_out_dim_wn'], 1).to(self.config['device'])

            self.q_t_n = nn.Linear(config['nenn_edge_out_dim_we']+config['nenn_node_out_dim_wn'], 1).to(self.config['device'])
            self.q_t_e = nn.Linear(2*config['nenn_edge_out_dim_we'], 1).to(self.config['device']).to(self.config['device'])

        else:
            self.w_n = nn.Linear( 2*config['nenn_node_out_dim_wn'], config['nenn_node_out_dim_wn']).to(self.config['device'])
            self.w_e = nn.Linear( 2*config['nenn_edge_out_dim_we'], config['nenn_edge_out_dim_we']).to(self.config['device'])
            
            self.a_t_n = nn.Linear(2*config['nenn_node_out_dim_wn'], 1).to(self.config['device']).to(self.config['device'])
            self.a_t_e = nn.Linear(config['nenn_edge_out_dim_we']+config['nenn_node_out_dim_wn'], 1).to(self.config['device'])

            self.q_t_n = nn.Linear(config['nenn_edge_out_dim_we']+config['nenn_node_out_dim_wn'], 1).to(self.config['device'])
            self.q_t_e = nn.Linear(2*config['nenn_edge_out_dim_we'], 1).to(self.config['device']).to(self.config['device'])

    def get_node_based_embedding_for_node(self, node_index, node_feature_matrix, adj_matrix ):
        
        num_rows = int(torch.sum(adj_matrix[node_index]).item())

        indices = torch.where(adj_matrix[node_index]>0)[0]
        
        
        dim_features = node_feature_matrix.shape[-1]
        
        required_tensor = torch.zeros(num_rows, 2*dim_features, dtype=node_feature_matrix.dtype, 
                                      device=self.config['device'])
        
        required_tensor[:,dim_features:] = node_feature_matrix[ node_index,:]
        
        for i, index in enumerate(indices):
            required_tensor[i, :dim_features] = node_feature_matrix[i,:]
        
        required_activations = F.leaky_relu(self.a_t_n(required_tensor))
        attention = F.softmax(required_activations, dim=0)
        
        x_ni = F.relu(torch.sum( required_tensor[:,:dim_features] * attention, dim=0 ))

        return x_ni

    def get_edge_based_embedding_for_node(self, node_index, edge_index,
                                          node_feature, edge_feature_matrix):
        '''
        edge_feature_matrix : 
        '''

        dim_edge_features = edge_feature_matrix.shape[-1]
        dim_node_features = node_feature.shape[0]

        neighbors1 = edge_index[0,:] == node_index
        neighbors2 = edge_index[1,:] == node_index

        neighbors = neighbors1 + neighbors2
        neighbor_indices = torch.where(neighbors > 0)[0]
        
        num_edge_based_neighbors = neighbor_indices.shape[0]
        
        required_tensor = torch.zeros(num_edge_based_neighbors, dim_edge_features+dim_node_features,
                                      dtype=node_feature.dtype, device=self.config['device'])
        
        required_tensor[:,dim_edge_features:] = node_feature
        
        for n in range(num_edge_based_neighbors):
            neighbor_index = neighbor_indices[n]
            temp_node0, temp_node1 = edge_index[:,neighbor_index]
            neighbor_feature = edge_feature_matrix[temp_node0, temp_node1, :]
            
            required_tensor[n, :dim_edge_features] = neighbor_feature

        required_activations = F.leaky_relu(self.a_t_e(required_tensor))
        attention = F.softmax(required_activations, dim=0)

        x_ei = F.relu(torch.sum( required_tensor[:,:dim_edge_features] * attention, dim=0 ))

        return x_ei


    def get_node_based_embedding_for_edge(self, edge_endpoints, edge_feature, 
                                          node_feature_matrix ):
        node0, node1 = int(edge_endpoints[0]), int(edge_endpoints[1])
        
        dim_edge_features = edge_feature.shape[0]
        dim_node_features = node_feature_matrix.shape[-1]
        
        required_tensor = torch.zeros(2, dim_edge_features+dim_node_features,
                                      dtype=edge_feature.dtype, device=self.config['device'])

        required_tensor[:,dim_node_features:] = edge_feature

        required_tensor[0,:dim_node_features] = node_feature_matrix[node0]
        required_tensor[1,:dim_node_features] = node_feature_matrix[node1]

        required_activations = F.leaky_relu(self.q_t_n(required_tensor))
        attention = F.softmax(required_activations, dim=0)

        e_ei = F.relu(torch.sum( required_tensor[:,:dim_node_features] * attention, dim=0 ))

        return e_ei
    
        
    def get_edge_based_embedding_for_edge(self, edge_pos, edge_index, edge_feature_matrix,
                                          line_adj_matrix):

        num_rows = int(torch.sum(line_adj_matrix[edge_pos]).item())
        indices = torch.where(line_adj_matrix[edge_pos]>0)[0]
        node0, node1 = edge_index[:,edge_pos]
        
        dim_features = edge_feature_matrix.shape[-1]
        
        required_tensor = torch.zeros(num_rows, 2*dim_features, dtype=edge_feature_matrix.dtype, 
                                      device=self.config['device'])
        
        required_tensor[:,dim_features:] = edge_feature_matrix[ node0, node1,:]
        
        for i, index in enumerate(indices):
            temp_node0, temp_node1 = edge_index[:,index]
            required_tensor[i, :dim_features] = edge_feature_matrix[temp_node0, temp_node1,:]
        
        required_activations = F.leaky_relu(self.q_t_e(required_tensor))
        attention = F.softmax(required_activations, dim=0)
        
        e_ei = F.relu(torch.sum( required_tensor[:,:dim_features] * attention, dim=0 ))

        return e_ei


    def get_concatenated_embeddings_of_node(self, node_index, edge_index, node_feature_matrix,
                                            edge_feature_matrix, adj_matrix):

        x_ni = self.get_node_based_embedding_for_node(node_index, node_feature_matrix, adj_matrix)
        node_feature = node_feature_matrix[node_index]
        
        x_ei = self.get_edge_based_embedding_for_node(node_index, edge_index, 
                                                      node_feature, edge_feature_matrix )
        x_i = torch.cat((x_ni, x_ei))
        return x_i
                                            

    def get_concatenated_embeddings_of_edge(self, edge_pos, edge_index, node_feature_matrix,
                                            edge_feature_matrix, line_adj_matrix):

        edge_endpoints = edge_index[:, edge_pos]
        node0, node1 = edge_endpoints
        
        edge_feature = edge_feature_matrix[node0, node1, :]
        
        e_ni = self.get_node_based_embedding_for_edge(edge_endpoints, edge_feature, node_feature_matrix )
        
        e_ei = self.get_edge_based_embedding_for_edge(edge_pos, edge_index, edge_feature_matrix, line_adj_matrix)
        e_i = torch.cat((e_ni, e_ei))
        
        return e_i


    def forward(self, node_features, edge_index, line_adj_matrix, adj_matrix, edge_features):
        '''
        forward function:

        node_features is the input feature vector size [num_nodes, node_feature_dimension]
        edge_index is the edge index vector, which has been curated for NENN use, different from pytorch 
        geometric one - it has been sent by cutting off the padded part
        line_adj_matrix is the adjacency matrix for line graph
        adj_matrix is the adjacency matrix of nodes
        relative features is a tensor of size [11, num_nodes, num_nodes, 7]
        '''

        W_n = self.w_n(node_features)
        W_e = self.w_e(edge_features)
        
        num_nodes = int(node_features.shape[0])
        num_edges = int(edge_index.shape[1])

        node_set = [i for i in range(num_nodes)]
        edge_set = [i for i in range(num_edges)]

        node_feature_dimension = self.config['nenn_node_out_dim_wn'] + self.config['nenn_edge_out_dim_we']
        edge_feature_dimension = self.config['nenn_edge_out_dim_we'] + self.config['nenn_node_out_dim_wn'] 
        
        node_embeddings = torch.zeros((num_nodes, node_feature_dimension)).to(self.config['device'])
        edge_embeddings = torch.zeros((num_edges, edge_feature_dimension)).to(self.config['device'])
        
        for n in node_set:
            node_embeddings[n] = self.get_concatenated_embeddings_of_node(n, edge_index, W_n, W_e, adj_matrix)

        for e in edge_set:
            edge_embeddings[e] = self.get_concatenated_embeddings_of_edge(e, edge_index, W_n, W_e, 
                                                                          line_adj_matrix)

        size_edge_features = edge_features.shape[0]
        result_edge_embeddings = torch.zeros(size_edge_features, size_edge_features, edge_feature_dimension).to(self.config['device'])
        
        for e in edge_set:
            node0, node1 = edge_index[0,e], edge_index[1,e]
            result_edge_embeddings[node0, node1] = edge_embeddings[e]
            result_edge_embeddings[node1, node0] = edge_embeddings[e]
        
        return node_embeddings, result_edge_embeddings
