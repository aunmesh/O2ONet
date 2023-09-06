import os

import torch
import torch.nn as nn
import torch.autograd

from utils.utils import collate_edge_indices, collate_node_features, decollate_node_embeddings

from model.nn_modules.gnn import GNN
from model.nn_modules.relation_classifier_2 import relation_classifier_2
from utils.utils import aggregate


class mfurln(torch.nn.Module):

    def __init__(self, config):

        super(mfurln, self).__init__()

        self.config = config.copy()      
        self.classifier_input_dimension = self.config['node_gnn_dimensions'][-1]
        
        h_size = 128
        
        self.nfs = config['node_feature_size']
        self.efs = config['edge_feature_size']
        
        self.indeterminate_fc_1 = self.make_linear_layer([self.nfs + self.efs, h_size])
        self.indeterminate_fc_2 = self.make_linear_layer([h_size, 1])
        
        h_size_2 = 512
        self.relationdet_fc_1 = self.make_linear_layer([self.nfs + self.efs + h_size, h_size_2])
        self.cr_cls = self.make_linear_layer([h_size_2, 3])
        self.lr_cls = self.make_linear_layer([h_size_2, 5])
        self.mr_cls = self.make_linear_layer([h_size_2, 14])
        


        # Hyperparameters to process node embeddings for classification
        self.agg = self.config['aggregator']
        
        self.label_locating_matrix = -1 * torch.ones(8, 8, dtype=torch.long, 
                                                 device=self.config['device']
                                                )
        curr_pos = 0
        for i in range(8):
            for j in range(i+1, 8):
                self.label_locating_matrix[i,j] = curr_pos
                self.label_locating_matrix[j,i] = curr_pos
                curr_pos+=1


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

    def make_all_pairs(self, data_item):
        
        batch_size = data_item['num_relation'].shape[0]
        
        feature_matrix = torch.zeros(batch_size, 28,
                                     self.nfs + self.efs, 
                                     device=self.config['device'])
        
        for b in range(batch_size):
            
            curr_num_obj = data_item['num_obj'][b]
            
            curr_index = 0
            
            for i in range(curr_num_obj):
                for j in range(i+1,curr_num_obj):
                    
                    node_feat_1 = data_item['concatenated_node_features'][b, i]
                    node_feat_2 = data_item['concatenated_node_features'][b, j]
                    
                    temp_node_feat = (node_feat_1 + node_feat_2)/2.0
                    temp_edge_feat = data_item['interaction_feature'][b, i,j]
                    
                    feature_matrix[b, curr_index, :self.nfs] = temp_node_feat
                    feature_matrix[b, curr_index, self.nfs:] = temp_edge_feat
        
        label_tensor = torch.zeros(batch_size, 28, device=self.config['device'])
        
        for b in range(batch_size):
            num_rels = data_item['num_relation'][b]
            indices0 = data_item['object_pairs'][:num_rels, 0]
            indices1 = data_item['object_pairs'][:num_rels, 1]
            
            pos_indices = self.label_locating_matrix[indices0, indices1]
            label_tensor[b, pos_indices] = 1
        return feature_matrix, label_tensor
                    
        
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
        
        # Determinate confidence subnetwork
        
        # Make all the possible pairs and put the predictions
        
        feature_matrix, label_tensor = self.make_all_pairs(data_item)
        
        indeterminate_1 = self.indeterminate_fc_1(feature_matrix)
        indeterminate_output = torch.sigmoid(self.indeterminate_fc_2(indeterminate_1)).squeeze()
        
        concatenated_features = torch.cat([feature_matrix, indeterminate_1], dim=-1)
        concatenated_features = self.relationdet_fc_1(concatenated_features)
        pos_features, neg_features = self.separate_the_feats(concatenated_features, data_item)
        
        
        # Now make two separate splits of concatenated features
        # One with annotated outputs
        # One without annotated outputs

        predictions = {}
        predictions['indeterminate_out'] = indeterminate_output
        predictions['label_tensor'] = label_tensor
        
        predictions['combined'] = {}

        # Make the batch for features
        predictions['combined']['lr'] = self.lr_cls(pos_features)
        predictions['combined']['cr'] = self.cr_cls(pos_features)
        predictions['combined']['mr'] = self.mr_cls(pos_features)
        
        predictions['lr_neg'] = self.lr_cls(neg_features)
        predictions['cr_neg'] = self.cr_cls(neg_features)
        predictions['mr_neg'] = self.mr_cls(neg_features)
        
        return predictions
    
    def separate_the_feats(self, concatenated_features, data_item):
        
        req_dim = concatenated_features.shape[-1]
        num_pairs = 8
        batch_size = concatenated_features.shape[0]
        
        pos_features = torch.zeros((batch_size,num_pairs, req_dim), device=self.config['device'])
        pairs = data_item['object_pairs']
        
        total_num_pos_relation = torch.sum(data_item['num_relation'])
        total_num_neg_relation = 28*batch_size - total_num_pos_relation
        
        neg_features = torch.zeros((total_num_neg_relation, req_dim), device=self.config['device'])
        
        lower_neg=0
        all_locs = [i for i in range(28)]
        for b in range(batch_size):
            pos_locs = []
            temp_num_pairs = int(data_item['num_relation'][b])
            for i in range(temp_num_pairs):

                ind0, ind1 = pairs[b, i, 0], pairs[b, i, 1]
                loc = int(self.label_locating_matrix[ind0, ind1].item())
                pos_locs.append(loc)
                emb = concatenated_features[b, loc]
                pos_features[b, i, :] = emb
            
            temp_neg_locs = list(set(all_locs)-set(pos_locs))
            num_neg = len(temp_neg_locs)
            temp_neg_locs = torch.tensor(temp_neg_locs)
            
            neg_features[lower_neg:lower_neg+num_neg,:] = concatenated_features[b, temp_neg_locs]
            lower_neg+=num_neg
        
        
        return pos_features, neg_features
                
             
        

        
        
        
        
        
