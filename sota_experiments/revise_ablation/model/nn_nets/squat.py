import os

import torch
import torch.nn
import torch.autograd

from model.nn_modules.relation_classifier import relation_classifier

from utils.utils import aggregate
from model.nn_modules.squat import SquatContext
class SQUAT(torch.nn.Module):

    def __init__(self, config):

        super(SQUAT, self).__init__()
        self.config = config.copy()
        self.squat_context = SquatContext(self.config).to(self.config['device'])
        
        if self.config['resize_feature_to_message_size']:
            # Resize large features

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

            config['edge_feature_size'] = config['message_size']
            config['node_feature_size'] = config['message_size']


        # creating the cr classifier
        cr_dim = self.config['cr_dimensions']
        scr_dropout = self.config['cr_dropout']
        self.cr_cls = relation_classifier(
            cr_dim, scr_dropout, self.config['device'], 1)
        self.cr_softmax = torch.nn.Softmax(dim=-1)

        # creating the lr classifier
        lr_dim = self.config['lr_dimensions']
        lr_dropout = self.config['lr_dropout']
        self.lr_cls = relation_classifier(
            lr_dim, lr_dropout, self.config['device'], 1)

        # creating the mr classifier
        mr_dim = self.config['mr_dimensions']
        mr_dropout = self.config['mr_dropout']
        self.mr_cls = relation_classifier(
            mr_dim, mr_dropout, self.config['device'], 1)

        # Hyperparameters to process node embeddings for classification
        self.agg = self.config['aggregator']
        self.classifier_input_dimension = self.config['message_size']


    def make_classifier_inputs(self, node_features, edge_features, edge_embeddings, pairs, num_rels):
        
        # Algorithm:
        # Goal: create a tensor which has shape b_size, max_num_pairs, feat_dim
        # each row has to correspond to the given pair of objects
        # using the pair of objects we can get the features
        
        feat_dim = 3 * edge_embeddings.shape[-1]
        num_pairs = pairs.shape[1]   # Always equal to max pairs
        num_batches = edge_embeddings.shape[0]


        classifier_input = torch.zeros(
                                        num_batches, num_pairs, 
                                        feat_dim, 
                                        device=self.config['device']
                                       )
        
        for b in range(num_batches):
            
            temp_num_pairs = num_rels[b]
            
            for i in range(temp_num_pairs):

                ind0, ind1 = pairs[b, i, 0], pairs[b, i, 1]
                
                n1, n2 = node_features[b, ind0, :], node_features[b, ind1, :]
                classifier_input[b, i, : 2*edge_embeddings.shape[-1] ] = aggregate(n1, n2, 'concat')
                
                emb0 = edge_embeddings[b, ind0, ind1]
                emb1 = edge_embeddings[b, ind1, ind0]
                temp_e_0 = aggregate(emb0, emb1, 'mean')
                
                emb0 = edge_features[b, ind0, ind1]
                emb1 = edge_features[b, ind1, ind0]
                temp_e_1 = aggregate(emb0, emb1, 'mean')	
                
                temp_e = aggregate(temp_e_0, temp_e_1, 'mean')
                
                classifier_input[b, i, 2*edge_embeddings.shape[-1]: ] = temp_e
                
                
        return classifier_input, num_pairs


    def forward(self, data_item):

        edge_features = data_item['interaction_feature']
        node_features = data_item['concatenated_node_features']

        batch_size = node_features.size()[0]

        edge_features = self.edge_feature_resize(edge_features)
        node_features = self.node_feature_resize(node_features)

        
        
        edge_embeddings, pred_masks = self.squat_context(node_features, edge_features,
                                                                            data_item['num_obj'])
        
        reshaped_masks = []

        for mask_list in pred_masks:
            
            new_list = []
            
            for m in mask_list:
            
                curr_len = m.shape[0]
                temp = int(curr_len ** 0.5)
            
                temp_mask = m.view(temp, temp)
                new_tensor = torch.zeros(12, 12, device=m.device)
            
                new_tensor[:temp, :temp] = temp_mask
                new_list.append(new_tensor.unsqueeze(0))
            
            reshaped_masks.append(torch.cat(new_list, dim=0))
                
                
        classifier_input, num_pairs = self.make_classifier_inputs(node_features, edge_features,
                                                                  edge_embeddings, 
                                                                  data_item['object_pairs'],
                                                                  data_item['num_relation'])

        predictions = {}
        predictions['combined'] = {}

        # Make the batch for features
        predictions['combined']['lr'] = self.lr_cls(num_pairs, classifier_input, 
                                                    batch_size)

        predictions['combined']['cr'] = self.cr_cls(num_pairs, classifier_input, batch_size)
        
        predictions['combined']['mr'] = self.mr_cls(num_pairs, classifier_input, 
                                                    batch_size)
        predictions['combined']['masks'] = reshaped_masks


        return predictions