import torch.nn as nn
import torch.nn.functional as F
import torch

class relation_classifier(nn.Module):
    '''
    classifier for classifying static contact relations
    '''

    def __init__(self, dimensions, dropout, device, flag=0):
        '''
        creates a multi-layered classifier
        dimensions are a list for [input dimension, hidden....hidden, output dimension]
        drouput is the dropout probability. Same for all layers
        '''

        super(relation_classifier, self).__init__()

        self.layers = nn.ModuleList()  # for storing all the transform layers
        self.dimensions = dimensions
        self.dropout = dropout
        self.device = device
        self.flag = flag

        for i in range(len(self.dimensions) - 1):

            curr_d = dimensions[i]
            next_d = dimensions[i+1]

            temp_fc_layer = nn.Linear(curr_d, next_d)
            self.layers.append(temp_fc_layer)

    def forward(self, num_pairs, classifier_input, num_batches):
        '''
        node_embeddings: node_embeddings of the graph
        pairs: pairs between which we have to classify the relations
        '''
        if self.flag:
            result_tensor = torch.empty( num_batches, num_pairs, 
                                        self.dimensions[-1], device=self.device).double()
        else:
            result_tensor = torch.empty(num_batches, num_pairs, 
                                        self.dimensions[-1], device=self.device)

        for b in range(num_batches):
            for i in range(num_pairs):
                x = classifier_input[b, i]
                for j in range(len(self.layers)-1):
                    x = F.relu(self.layers[j](x))
                    x = F.dropout(x, self.dropout)

                result_tensor[b, i] = self.layers[-1](x)

        return result_tensor