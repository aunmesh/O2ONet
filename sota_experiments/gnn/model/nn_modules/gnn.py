import torch.nn as nn
import torch.nn.functional as F
from module_utils import get_gnn
import torch

class GNN(nn.Module):

    def __init__(self, config, dimensions, dropout):
        '''
        creates a graph neural network
        dimensions are a list for [input dimension, hidden....hidden, output dimension]
        drouput is the dropout probability. Same for all layers
        '''

        super(GNN, self).__init__()

        self.gc_layers = nn.ModuleList()  # for storing all the Graphical Convolution Layers
        self.dimensions = dimensions
        self.dropout = dropout

        for i in range(len(self.dimensions) - 1):
            
            curr_d = dimensions[i]
            next_d = dimensions[i+1]

            temp_gc_layer = get_gnn(curr_d, next_d)
            self.gc_layers.append(temp_gc_layer)

    def forward(self, x, edge_index):
        '''
        forward function where x is the input feature vector
        and edge_index is the Adjacency matrix

        forward pass can be represented as Dropout(Relu(GC(x)))
        where GC is Graph Convolution
        '''
        for i in range(len(self.gc_layers)):
            x = F.relu(self.gc_layers[i](x, edge_index))
            x = F.dropout(x, self.dropout)
        return x
