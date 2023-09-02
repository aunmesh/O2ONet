import os

import torch
import torch.nn as nn
import torch.autograd
from utils.utils import collate_edge_indices, collate_node_features, decollate_node_embeddings
from model.nn_modules.gnn import GNN
from model.nn_modules.relation_classifier_2 import relation_classifier_2
from utils.utils import aggregate

class graph_rcnn(torch.nn.Module):

    def __init__(self, config):

        super(graph_rcnn, self).__init__()
        self.config = config.copy()
        
        self.rel_pn_subject = self.make_linear_layer(self.config['rel_pn_dimensions'])
        self.rel_pn_object = self.make_linear_layer(self.config['rel_pn_dimensions'])
        
        self.a_gcn = GNN(self.config)
        self.classifier_input_dimension = self.config['gnn_dimensions'][-1]
        
        # creating the cr classifier
        cr_dim = self.config['cr_dimensions']
        scr_dropout = self.config['cr_dropout']
        self.cr_cls = relation_classifier_2(
            cr_dim, scr_dropout, self.config['device'], 1)
        self.cr_softmax = torch.nn.Softmax(dim=-1)

        # creating the lr classifier
        lr_dim = self.config['lr_dimensions']
        lr_dropout = self.config['lr_dropout']
        self.lr_cls = relation_classifier_2(
            lr_dim, lr_dropout, self.config['device'], 1)

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
            num_batches, num_pairs, self.classifier_input_dimension, device=self.config['device'])

        for b in range(num_batches):

            for i in range(num_pairs):

                ind0, ind1 = pairs[b, i, 0], pairs[b, i, 1]

                emb0, emb1 = node_embeddings[b, ind0], node_embeddings[b, ind1]
                classifier_input[b, i] = aggregate(emb0, emb1, self.agg)

        return classifier_input


    def forward(self, data_item):
        """
        data_item - 
        Run the relationship proposal network
        Threshold on the score and get the edges.
        Add the edges to the graph which need to be classified
        Construct the relationship + object graph
        Send the graph to the agcn
        Construct the classification matrix
        Classify
        
        data_item keys:
        ['metadata', 'geometric_feature', 'appearance_feature', 'i3d_feature',
        'motion_feature', 'num_obj', 'edge_index', 'num_edges', 'object_pairs',
        'scr', 'lr', 'mr', 'obj_mask', 'obj_pairs_mask', 'num_pairs', 'scr_metric', 
        'lr_metric', 'mr_metric', 'relative_feature', 'nenn_edge_index', 'nenn_num_edges', 
        'line_adj_mat', 'adj_mat']
        
        + ['concatenated_node_features', 'relative_spatial_feature']
        """
        batch_size = data_item['concatenated_node_features'].shape[0]
        
        # Relation Proposal Network Phase
        phi_embedding = self.rel_pn_subject( data_item['concatenated_node_features'] )
        psi_embedding = self.rel_pn_object( data_item['concatenated_node_features'] )
        relatedness_score = torch.sigmoid( torch.einsum('bik,bjk->bij', phi_embedding, psi_embedding) )
        
        # Threshold on the score and get the edges
        rel_pn_output = (relatedness_score > 0.5)
        
        # Add the edges to the graph which need to be classified
        for b in range(batch_size):
            # Masking the objects which are not present
            temp_num_obj = data_item['num_obj'][b]
            # print("DEBUG", rel_pn_output.shape)
            rel_pn_output[b,temp_num_obj:,:] = False
            rel_pn_output[b,:,temp_num_obj:] = False

            # Adding the relations which need to be classified
            temp_num_relations = data_item['num_edges'][b]
            related_pairs = data_item['object_pairs'][b, :temp_num_relations]
            
            num_related_pairs = related_pairs.shape[0]
            
            for n in range(num_related_pairs):
                index0, index1 = related_pairs[n]
                
                rel_pn_output[b, index0, index1] = True
                rel_pn_output[b, index1, index0] = True            
        
        # Constructing the graph. The features for the 
        # relationship node need to be the relative feature
        '''
        How do we construct the graph. We definitely need to add edges and construct the graph.
        Let's skip adding the relationship nodes for now and see the results.
        
        Do we skip adding the relationship nodes at all?
        That wouldn't be fair obviously.
        On the other hand we can complete the pipeline fast and see it running.
        '''
        
        # Let's construct the appropriate edge indices
        max_num_edges = 172
        edge_indices = -1 * torch.ones(batch_size, 2, max_num_edges, 
                                  dtype=torch.long, device = self.config['device'])
        
        num_edges_for_batch = []
        for b in range(batch_size):
            curr_adj_mat = rel_pn_output[b]
            
            indices_start, indices_end = torch.where(curr_adj_mat == True)
            
            temp_num_edges = len(indices_start)
            num_edges_for_batch.append(temp_num_edges)
            
            edge_indices[b, 0, :temp_num_edges] = indices_start
            edge_indices[b, 1, :temp_num_edges] = indices_end
        
        num_edges_for_batch = torch.tensor(num_edges_for_batch, device=self.config['device'])
        # Now since the edge indices are calculated we need 
        # to collate the edge indices and node features.
        
        # collate_edge_indices(edge_index, num_edges, num_objects, device)
        
        collated_edge_index, edge_slicing = collate_edge_indices(edge_indices, num_edges_for_batch, 
                             data_item['num_obj'], self.config['device'])
        
        collated_node_feature, node_slicing = collate_node_features(data_item['concatenated_node_features'],
                                                       data_item['num_obj'], self.config['device']
                                                       )
        
        
       
        
        all_node_embeddings = self.a_gcn(collated_node_feature, collated_edge_index)
        node_embeddings = decollate_node_embeddings( all_node_embeddings, node_slicing, 
                                                    self.config['device'])


        classifier_input = self.make_classifier_inputs(node_embeddings, data_item['object_pairs'])

        predictions = {}
        predictions['combined'] = {}
        # Make the batch for features
        predictions['combined']['lr'] = self.lr_cls(classifier_input)
        
        predictions['combined']['cr'] = self.cr_softmax( self.cr_cls(classifier_input) )
        
        predictions['combined']['mr'] = self.mr_cls(classifier_input)

        return predictions