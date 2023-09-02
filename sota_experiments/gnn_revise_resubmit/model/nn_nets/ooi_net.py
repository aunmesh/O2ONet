from utils.utils import decollate_node_embeddings

from model.nn_modules.relation_classifier import relation_classifier
from model.nn_modules.gnn import GNN 

from utils.utils import aggregate

import torch.nn as nn
import torch.nn.functional as F
import torch


class ooi_net(nn.Module):
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
        super(ooi_net, self).__init__()
        self.config = config

        # creating the gcn
        self.device = config['device']
        
        # Creating the Graph Neural Network
        gcn_dim = config['gcn_dimensions']
        gcn_dropout = config['gcn_dropout']
        self.GNN = GNN(self.config, gcn_dim, gcn_dropout).double()
        
        # creating the relation classifiers

        # creating the scr classifier
        cr_dim = config['cr_dimensions']
        scr_dropout = config['cr_dropout']
        self.cr_cls = relation_classifier(cr_dim, scr_dropout, self.device)

        # creating the mr classifier
        lr_dim = config['lr_dimensions']
        lr_dropout = config['lr_dropout']
        self.lr_cls = relation_classifier(lr_dim, lr_dropout, self.device)

        # creating the lr classifier
        mr_dim = config['mr_dimensions']
        mr_dropout = config['mr_dropout']
        self.mr_cls = relation_classifier(mr_dim, mr_dropout, self.device)
        
        
        # Hyperparameters to process node embeddings for classification
        self.agg = config['aggregator']
        self.classifier_input_dimension = gcn_dim[-1]


    def make_classifier_inputs(self, node_embeddings, pairs):

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

                emb0, emb1 = node_embeddings[b, ind0], node_embeddings[b, ind1]
                classifier_input[b, i] = aggregate(emb0, emb1, self.agg)

        return num_pairs, classifier_input

    #def forward(self, obj_features, edge_index, obj_pairs, slicing):
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
        collated_obj_features = data_item['collated_obj_features']
        collated_edge_index = data_item['collated_edge_index']
        
        # convolving the obj_features
        all_obj_embeddings = self.GNN(collated_obj_features, collated_edge_index)
        
        
        obj_embeddings = decollate_node_embeddings(
            all_obj_embeddings, data_item['slicing']['node'], self.device)

        num_pairs, classifier_input = self.make_classifier_inputs(
            obj_embeddings, data_item['object_pairs'])

        num_batches = obj_embeddings.shape[0]

        predictions = {}

        # Make the batch for features
        predictions['lr'] = self.lr_cls(num_pairs, classifier_input, num_batches)
        predictions['cr'] = self.cr_cls(num_pairs, classifier_input, num_batches)
        predictions['mr'] = self.mr_cls(num_pairs, classifier_input, num_batches)

        return predictions