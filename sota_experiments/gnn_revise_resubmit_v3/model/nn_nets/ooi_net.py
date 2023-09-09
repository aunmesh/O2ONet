from utils.utils import decollate_node_embeddings
from utils.utils import collate_node_features, collate_edge_indices

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


        self.edge_feature_resize = torch.nn.Linear(
            config['edge_feature_size'],
            config['message_size']
        ).to(config['device'])

        self.node_feature_resize = torch.nn.Linear(
            config['node_feature_size'],
            config['message_size']
        ).to(config['device'])

        torch.nn.init.xavier_normal(self.edge_feature_resize.weight)
        torch.nn.init.xavier_normal(self.node_feature_resize.weight)


        # creating the gcn
        self.device = config['device']
        
        # Creating the Graph Neural Network
        gcn_dim = config['gnn_dimensions']
        gcn_dropout = config['gnn_dropout']
        self.GNN = GNN(self.config)
        
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

        self.classifier_input_dimension = gcn_dim[-1] + self.config['edge_feature_size']


    def make_classifier_inputs(self, obj_ft, node_embeddings, pairs, edge_embeddings):
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
            num_batches, num_pairs, 3* self.config['message_size'], device=self.device)

        for b in range(num_batches):

            for i in range(num_pairs):

                ind0, ind1 = pairs[b, i, 0], pairs[b, i, 1]

                edge_embedding_1 = edge_embeddings[b, ind0, ind1, :]
                edge_embedding_2 = edge_embeddings[b, ind1, ind0, :]
                
                temp_e = aggregate(edge_embedding_1, edge_embedding_2, 'mean')	

                classifier_input[b, i, 2 * self.config['message_size']:] = temp_e
                
                emb0, emb1 = node_embeddings[b, ind0], node_embeddings[b, ind1]
                temp_n1 = aggregate(emb0, emb1, 'concat')
                
                emb0, emb1 = obj_ft[b, ind0], obj_ft[b, ind1]
                temp_n2 = aggregate(emb0, emb1, 'concat')                
                
                temp_n = aggregate(temp_n1, temp_n2, 'mean')
                
                classifier_input[b, i, :2*self.config['message_size']] = temp_n


        return classifier_input


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
        obj_features = data_item['concatenated_node_features']
        edge_features = data_item['interaction_feature']
        
        obj_ft = self.node_feature_resize(obj_features)
        edge_ft = self.edge_feature_resize(edge_features)
        
        edge_index = data_item['edge_index']
        
        collated_obj_features, node_slices = collate_node_features(obj_features, data_item['num_obj'],
                                                      self.config['device'])
        
        collated_edge_index, edge_slices = collate_edge_indices(
                                                    edge_index, data_item['num_edges'],
                                                    data_item['num_obj'], self.config['device']
                                                   )
        
        
        # convolving the obj_features
        all_obj_embeddings = self.GNN(collated_obj_features, collated_edge_index)
        
        obj_embeddings = decollate_node_embeddings(
            all_obj_embeddings, node_slices, self.device)

        
        
        classifier_input = self.make_classifier_inputs( obj_ft,
            obj_embeddings, data_item['object_pairs'], edge_ft)
        
        num_pairs = data_item['object_pairs'].shape[1]
        
        num_batches = obj_embeddings.shape[0]

        predictions = {}
        predictions['combined'] = {}
        
        # Make the batch for features
        predictions['combined']['lr'] = self.lr_cls(num_pairs, classifier_input, num_batches)
        predictions['combined']['cr'] = self.cr_cls(num_pairs, classifier_input, num_batches)
        predictions['combined']['mr'] = self.mr_cls(num_pairs, classifier_input, num_batches)

        return predictions