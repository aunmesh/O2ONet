import os

import torch
import torch.nn as nn
import torch.autograd
from utils.utils import collate_edge_indices, collate_node_features
from nn_modules.gnn import GNN

class graph_rcnn(torch.nn.Module):

    def __init__(self, config):

        super(graph_rcnn, self).__init__()
        self.config = config.copy()
        
        self.rel_pn_subject = self.make_linear_layer(self.config['rel_pn_dimensions'])
        self.rel_pn_object = self.make_linear_layer(self.config['rel_pn_dimensions'])
        
        self.a_gcn = GNN(self.config)
        
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
        batch_size = data_item['concatenated_node_feature'].shape[0]
        
        # Relation Proposal Network Phase
        phi_embedding = self.rel_pn_subject( data_item['concatenated_node_features'] )
        psi_embedding = self.rel_pn_object( data_item['concatenated_node_features'] )
        relatedness_score = torch.sigmoid( torch.einsum('bik,bjk->bij', phi_embedding, psi_embedding) )
        
        # Threshold on the score and get the edges
        rel_pn_output = (relatedness_score > 0.5)
        
        # Add the edges to the graph which need to be classified
        for b in range(batch_size):
            # Masking the objects which are not present
            temp_num_obj = data_item['num_obj']
            rel_pn_output[b,temp_num_obj:,temp_num_obj:] = False

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
        max_num_edges = 128
        edge_indices = -1 * torch.ones(batch_size, 2, max_num_edges, 
                                  dtype=torch.long, device = self.config['device'])
        
        batch_num_edges = []
        for b in range(batch_size):
            curr_adj_mat = rel_pn_output[b]
            
            indices_start, indices_end = torch.where(curr_adj_mat)
            
            temp_num_edges = len(indices_start)
            batch_num_edges.append(temp_num_edges)
            
            edge_indices[b, 0, :temp_num_edges] = indices_start
            edge_indices[b, 1, :temp_num_edges] = indices_end
        
        batch_num_edges = torch.tensor(batch_num_edges, device=self.config['device'])
        
        # Now since the edge indices are calculated we need 
        # to collate the edge indices and node features.
        
        # collate_edge_indices(edge_index, num_edges, num_objects, device)
        
        collated_edge_index = collate_edge_indices(edge_indices, batch_num_edges, 
                             data_item['num_obj'], self.config['device'])
        
        collated_node_features = collate_node_features(data_item['concatenated_node_feature'],
                                                       data_item['num_obj'], self.config['device']
                                                       )
        
        output_node_features = self.a_GCN(collated_node_features, collated_edge_index)
        
        
        
        
        
        
        
        
        
        

        return 