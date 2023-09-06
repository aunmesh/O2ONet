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
        
        self.object_node_embedder = self.make_linear_layer([config['node_feature_size'], 512])
        self.relationship_node_embedder = self.make_linear_layer([config['edge_feature_size'], 512])
        
        self.a_gcn = GNN(self.config)
        self.classifier_input_dimension = self.config['gnn_dimensions'][-1]*2
        
        # creating the cr classifier
        cr_dim = self.config['cr_dimensions']
        scr_dropout = self.config['cr_dropout']
        self.cr_cls = relation_classifier_2(
            cr_dim, scr_dropout, self.config['device'], 1)

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


    def make_classifier_inputs_graph_rcnn(self, node_embeddings, 
                                          edge_embeddings, pairs, num_rels, edge_locating_tensor):
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
                                        self.classifier_input_dimension, 
                                        device=self.config['device']
                                       )
        
        half_dimension = self.classifier_input_dimension // 2
        for b in range(num_batches):
            
            edge_offset = torch.sum(num_rels[:b])
            temp_num_pairs = num_rels[b]
            
            for i in range(num_pairs):

                ind0, ind1 = pairs[b, i, 0], pairs[b, i, 1]
                emb0, emb1 = node_embeddings[b, ind0], node_embeddings[b, ind1]
                
                classifier_input[b, i, :half_dimension] = aggregate(emb0, emb1, self.agg)
                
                # to get the relationship embedding we need to get the 2 edges corresponding
                # to ind0 ind1 and ind1 ind0
                
                # We need to locate where is ind0 and ind1 and then we need to loca
                loc0 = (edge_locating_tensor[:,0] == b)
                #########
                loc1 = (edge_locating_tensor[:,1] == ind0)
                loc2 = (edge_locating_tensor[:,2] == ind1)
                
                embed_loc = torch.argmax(loc1*loc2*loc0*1.0).item()
                embed1 = edge_embeddings[embed_loc]


                #########
                loc1 = (edge_locating_tensor[:,1] == ind1)
                loc2 = (edge_locating_tensor[:,2] == ind0)
                
                embed_loc = torch.argmax(loc1*loc2*loc0*1.0).item()
                embed2 = edge_embeddings[embed_loc]
                
                
                classifier_input[b, i, half_dimension:] = aggregate(embed1, embed2, self.agg)
                # classifier_input[b, i, half_dimension:] = edge_embeddings[edge_offset + i]

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
            temp_num_relations = data_item['num_relation'][b]
            related_pairs = data_item['object_pairs'][b, :temp_num_relations]
            
            num_related_pairs = related_pairs.shape[0]
            
            for n in range(num_related_pairs):
                index0, index1 = related_pairs[n]
                
                rel_pn_output[b, index0, index1] = True
                rel_pn_output[b, index1, index0] = True            
        
        # Construct the relationship + object graph
        # The features for the relationship node need to be the relative feature
        
        # Let's construct the appropriate edge indices
        max_num_edges = 172
        edge_indices = -1 * torch.ones(batch_size, 2, max_num_edges, 
                                    dtype=torch.long, device = self.config['device'])
        
        num_edges_for_batch = []
        
        total_num_edges = torch.sum(rel_pn_output).item()
        edge_locating_tensor = -1 * torch.ones((total_num_edges, 3), device=self.config['device'])
        
        lower = 0
        for b in range(batch_size):
            curr_adj_mat = rel_pn_output[b]
            
            indices_start, indices_end = torch.where(curr_adj_mat == True)
            
            temp_num_edges = len(indices_start)
            num_edges_for_batch.append(temp_num_edges)
            
            edge_indices[b, 0, :temp_num_edges] = indices_start
            edge_indices[b, 1, :temp_num_edges] = indices_end
            
            edge_locating_tensor[lower:lower+temp_num_edges, 0] = b
            edge_locating_tensor[lower:lower+temp_num_edges, 1] = indices_start
            edge_locating_tensor[lower:lower+temp_num_edges, 2] = indices_end
        
        num_edges_for_batch = torch.tensor(num_edges_for_batch, device=self.config['device'])
        # Now since the edge indices are calculated we need 
        # to collate the edge indices and node features.
        
        # collate_edge_indices(edge_index, num_edges, num_objects, device)
        
        collated_edge_index, edge_slicing = collate_edge_indices(edge_indices, num_edges_for_batch, 
                             data_item['num_obj'], self.config['device'])
        
        collated_node_feature, node_slicing = collate_node_features(data_item['concatenated_node_features'],
                                                       data_item['num_obj'], self.config['device']
                                                       )
        
        
        # Now to this collated_node_feature and collated_edge_index we add the relationship graph
        """
        Steps:
        0. We embed both the relationship and object nodes in the same space.
        1. Relationship nodes are there for each batch item.
        2. We make a relationship node which is enumerated after the last object node.
        3. We find the mapping 
        """
        
        ## 0. We embed both the relationship and object nodes in the same space.
        collated_node_feature = self.object_node_embedder(collated_node_feature)
        
        # Getting the relationship node features.
        
        # total_num_rels = torch.sum( data_item['num_pairs'] )
        total_num_rels = torch.sum(num_edges_for_batch)
        
        rel_node_feats = torch.zeros(total_num_rels, self.config['edge_feature_size'], 
                                     device=self.config['device'])
        
        lower = 0
        
        for b in range(batch_size):
            # temp_num_rels = data_item['num_relation'][b]
            # temp_obj_pairs = data_item['object_pairs'][b, :temp_num_rels]

            # temp_indices_0 = temp_obj_pairs[:, 0]
            # temp_indices_1 = temp_obj_pairs[:, 1]

            # temp_num_rels = torch.sum(num_edges_for_batch == b)
            temp_num_rels = num_edges_for_batch[b].item()
            
            temp_indices_start = edge_indices[b, 0, :temp_num_rels]
            temp_indices_end = edge_indices[b, 1,  :temp_num_rels]
            # print("DEBUG 223", rel_node_feats[lower:lower+temp_num_rels, :].shape, 
            #       data_item['relative_spatial_feature'][b, temp_indices_start, temp_indices_end].shape)
            rel_node_feats[lower:lower+temp_num_rels, :] = data_item['interaction_feature'][b, temp_indices_start, temp_indices_end]
            
            lower += temp_num_rels
        
        rel_node_feats = self.relationship_node_embedder(rel_node_feats)
        
        
        rel_edge_indices = -1 * torch.ones(2, total_num_rels*4, dtype=torch.long, 
                                           device = self.config['device'])

        max_node_index = torch.max(collated_edge_index)
        
        lower = 0
        for b in range(batch_size):
            
            # temp_num_rels = data_item['num_relation'][b]
            # temp_obj_pairs = data_item['object_pairs'][b, :temp_num_rels]
            
            # offset = torch.sum(data_item['num_relation'][:b])
                                   
            # rel_node_enumeration = [ max_node_index + offset + i for i in range(temp_num_rels) ]
            # rel_node_enumeration = torch.tensor(rel_node_enumeration, dtype=torch.long,
            #                                     device = self.config['device'])



            temp_num_rels = num_edges_for_batch[b]

            temp_node_indices_start = edge_indices[b, 0,  :temp_num_rels]
            temp_node_indices_end = edge_indices[b, 1,  :temp_num_rels]

            offset = torch.sum(num_edges_for_batch[:b])

            rel_node_enumeration = [ max_node_index + offset + i for i in range(temp_num_rels) ]
            rel_node_enumeration = torch.tensor(rel_node_enumeration, dtype=torch.long,
                                                device = self.config['device'])

            # Need to use the slicing dictionary here
            # temp_indices_0 = offset + temp_obj_pairs[:, 0]
            # temp_indices_1 = offset + temp_obj_pairs[:, 1]

            # Need to use the slicing dictionary here
            node_offset = torch.argmin( 1.0*(node_slicing == b) ).item()
            
            temp_indices_0 = node_offset + temp_node_indices_start
            temp_indices_1 = node_offset + temp_node_indices_end
            
            rel_edge_indices[0, lower:lower+temp_num_rels] = temp_indices_0
            rel_edge_indices[1, lower:lower+temp_num_rels] = rel_node_enumeration
            
            rel_edge_indices[0, lower+temp_num_rels:lower+2*temp_num_rels] = rel_node_enumeration
            rel_edge_indices[1, lower+temp_num_rels:lower+2*temp_num_rels] = temp_indices_0

            rel_edge_indices[0, lower+2*temp_num_rels:lower+3*temp_num_rels] = temp_indices_1
            rel_edge_indices[1, lower+2*temp_num_rels:lower+3*temp_num_rels] = rel_node_enumeration
            
            rel_edge_indices[0, lower+3*temp_num_rels:lower+4*temp_num_rels] = rel_node_enumeration
            rel_edge_indices[1, lower+3*temp_num_rels:lower+4*temp_num_rels] = temp_indices_1
            
            lower += (temp_num_rels*4)


        # print("DEBUG", collated_node_feature.shape, rel_node_feats.shape)
        # print("DEBUG", collated_edge_index.shape, rel_edge_indices.shape)        

        node_and_rel_feats = torch.cat([collated_node_feature, rel_node_feats], dim=0)
        joint_edge_indices = torch.cat([collated_edge_index, rel_edge_indices], dim=-1)

        # print("DEBUG 2", node_and_rel_feats.shape)
        # print("DEBUG 2", joint_edge_indices.shape)        


        # This gives me the total graph appropriately connected to all the nodes
        # all_node_embeddings = self.a_gcn(collated_node_feature, collated_edge_index)
        
        node_and_rel_embeddings = self.a_gcn(node_and_rel_feats, joint_edge_indices)
        
        node_and_rel_embeddings_node_only = node_and_rel_embeddings[ :max_node_index + 1]
        node_and_rel_embeddings_edge_only = node_and_rel_embeddings[max_node_index + 1:]
        
        node_and_rel_embeddings_node_only = decollate_node_embeddings( node_and_rel_embeddings_node_only, 
                                                    node_slicing,
                                                    self.config['device'])

        classifier_input = self.make_classifier_inputs_graph_rcnn(
                                                                node_and_rel_embeddings_node_only,
                                                                node_and_rel_embeddings_edge_only,
                                                                data_item['object_pairs'],
                                                                data_item['num_relation'],
                                                                edge_locating_tensor
                                                                  )

        predictions = {}
        predictions['combined'] = {}
        # Make the batch for features
        predictions['combined']['lr'] = self.lr_cls(classifier_input)
        
        predictions['combined']['cr'] = self.cr_cls(classifier_input)
        
        predictions['combined']['mr'] = self.mr_cls(classifier_input)
        predictions['rel_proposal'] = relatedness_score
        return predictions