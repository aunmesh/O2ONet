from torch import double, tensor
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import get_gnn

class GNN(nn.Module):

    def __init__(self, config, key=''):
        '''
        creates a graph neural network
        dimensions are a list for [input dimension, hidden....hidden, output dimension]
        drouput is the dropout probability. Same for all layers
        '''

        super(GNN, self).__init__()

        self.config = config
        self.gc_layers = nn.ModuleList()  # for storing all the Graph Convolution Layers
        self.dimensions = self.config[key + 'gnn_dimensions']
        self.dropout = self.config[key + 'gnn_dropout']
        

        for i in range(len(self.dimensions) - 1):
            
            curr_d = self.dimensions[i]
            next_d = self.dimensions[i+1]

            temp_gc_layer = get_gnn(self.config, curr_d, next_d)
            self.gc_layers.append(temp_gc_layer)
        self.gc_layers = self.gc_layers.to(self.config['device'])

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