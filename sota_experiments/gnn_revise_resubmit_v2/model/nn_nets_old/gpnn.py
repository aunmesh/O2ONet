import os

import torch
import torch.nn
import torch.autograd

from model.nn_modules.LinkFunction import LinkFunction
from model.nn_modules.MessageFunction import MessageFunction
from model.nn_modules.ReadoutFunction import ReadoutFunction
from model.nn_modules.UpdateFunction import UpdateFunction
from model.nn_modules.relation_classifier import relation_classifier

from utils.utils import aggregate

class GPNN(torch.nn.Module):

    def __init__(self, config):

        super(GPNN, self).__init__()
        self.config = config.copy()

        if self.config['resize_feature_to_message_size']:
            # Resize large features

            self.edge_feature_resize = torch.nn.Linear(
                config['edge_feature_size'],
                config['message_size']
            ).to(config['device']).double()

            self.node_feature_resize = torch.nn.Linear(
                config['node_feature_size'],
                config['message_size']
            ).to(config['device']).double()

            torch.nn.init.xavier_normal(self.edge_feature_resize.weight)
            torch.nn.init.xavier_normal(self.node_feature_resize.weight)

            config['edge_feature_size'] = config['message_size']
            config['node_feature_size'] = config['message_size']

        self.link_fun = LinkFunction('GraphConv', config).to(
            self.config['device']).double()
        self.sigmoid = torch.nn.Sigmoid().to(self.config['device']).double()
        self.message_fun = MessageFunction(
            'linear_concat_relu', config).to(self.config['device']).double()
        self.update_fun = UpdateFunction('gru', config).to(
            self.config['device']).double()

        self.propagate_layers = config['propagate_layers']

        self._load_link_fun(config)

        # creating the cr classifier
        cr_dim = self.config['cr_dimensions']
        scr_dropout = self.config['cr_dropout']
        self.cr_cls = relation_classifier(
            cr_dim, scr_dropout, self.config['device'], 1).double()
        self.cr_softmax = torch.nn.Softmax(dim=-1)

        # creating the lr classifier
        lr_dim = self.config['lr_dimensions']
        lr_dropout = self.config['lr_dropout']
        self.lr_cls = relation_classifier(
            lr_dim, lr_dropout, self.config['device'], 1).double()

        # creating the mr classifier
        mr_dim = self.config['mr_dimensions']
        mr_dropout = self.config['mr_dropout']
        self.mr_cls = relation_classifier(
            mr_dim, mr_dropout, self.config['device'], 1).double()

        # Hyperparameters to process node embeddings for classification
        self.agg = self.config['aggregator']
        self.classifier_input_dimension = self.config['message_size']

    def make_classifier_inputs(self, node_embeddings, pairs, context_embedding):
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

        # classifier_input is the tensor which will be passed to the fully connected classifier
        # for feature classification
        classifier_input = torch.zeros(num_batches, num_pairs,
                                       self.classifier_input_dimension,
                                       device=self.config['device']).double()

        for b in range(num_batches):
            # temp_context_embedding = context_embedding[b, :]
            
            for i in range(num_pairs):

                ind0, ind1 = pairs[b, i, 0], pairs[b, i, 1]

                emb0, emb1 = node_embeddings[b, ind0], node_embeddings[b, ind1]
                temp_agg = aggregate(emb0, emb1, self.agg)
                classifier_input[b, i] = temp_agg
                # classifier_input[b, i] = aggregate(temp_agg, temp_context_embedding, 'concat')

        return num_pairs, classifier_input

    def forward(self, data_item):

        edge_features = data_item['relative_spatial_feature']
        node_features = data_item['concatenated_node_features']

        batch_size = node_features.size()[0]

        # maximum number of nodes in a gif
        max_num_nodes = data_item['concatenated_node_features'].size()[1]

        if self.config['resize_feature_to_message_size']:
            edge_features = self.edge_feature_resize(edge_features)
            node_features = self.node_feature_resize(node_features)

        edge_features = edge_features.permute(0, 3, 1, 2)
        node_features = node_features.permute(0, 2, 1)

        hidden_node_states = [[node_features[batch_i, ...].unsqueeze(0).clone() for _ in range(
            self.propagate_layers+1)] for batch_i in range(node_features.size()[0])]
        hidden_edge_states = [[edge_features[batch_i, ...].unsqueeze(0).clone() for _ in range(
            self.propagate_layers+1)] for batch_i in range(node_features.size()[0])]

        # pred_adj_mat = torch.autograd.Variable(torch.zeros(adj_mat.size()))
        pred_adj_mat = torch.autograd.Variable(
            torch.zeros(batch_size, max_num_nodes, max_num_nodes))

        # pred_node_labels = torch.autograd.Variable(torch.zeros(node_labels.size()))

        if self.config['device'].type == 'cuda':
            pred_adj_mat = pred_adj_mat.to(self.config['device'])

        for batch_idx in range(node_features.size()[0]):

            #valid_node_num = human_nums[batch_idx] + obj_nums[batch_idx]
            valid_node_num = data_item['num_obj'][batch_idx]

            for passing_round in range(self.propagate_layers):

                pred_adj_mat[batch_idx, :valid_node_num, :valid_node_num] = self.link_fun(
                    hidden_edge_states[batch_idx][passing_round][:, :, :valid_node_num, :valid_node_num])
                sigmoid_pred_adj_mat = self.sigmoid(
                    pred_adj_mat[batch_idx, :, :]).unsqueeze(0)

                # Loop through nodes
                for i_node in range(valid_node_num):
                    h_v = hidden_node_states[batch_idx][passing_round][:, :, i_node]
                    h_w = hidden_node_states[batch_idx][passing_round][:,
                                                                       :, :valid_node_num]
                    e_vw = edge_features[batch_idx, :,
                                         i_node, :valid_node_num].unsqueeze(0)
                    m_v = self.message_fun(h_v, h_w, e_vw, self.config)

                    # Sum up messages from different nodes according to weights
                    m_v = sigmoid_pred_adj_mat[:, i_node, :valid_node_num].unsqueeze(
                        1).expand_as(m_v) * m_v
                    hidden_edge_states[batch_idx][passing_round +
                                                  1][:, :, :valid_node_num, i_node] = m_v
                    m_v = torch.sum(m_v, 2)

                    h_v = self.update_fun(h_v[None].contiguous(), m_v[None])

                    # might need to change it for videos
                    hidden_node_states[batch_idx][passing_round +
                                                  1][0, :, i_node] = h_v[0][0]

        final_embeddings = [hidden_node_states[batch_idx]
                            [-1].permute(0, 2, 1) for batch_idx in range(batch_size)]
        final_embeddings = torch.cat(final_embeddings, 0)
        
        # print("DEBUG 6", final_embeddings.size())
        
        num_pairs, classifier_input = self.make_classifier_inputs(
            final_embeddings,
            data_item['object_pairs'],
            data_item['activity_embedding']
        )

        predictions = {}
        predictions['combined'] = {}
        # Make the batch for features
        predictions['combined']['lr'] = self.lr_cls(num_pairs, classifier_input, 
                                                    batch_size)
        
        # predictions['combined']['cr'] = self.cr_softmax( self.cr_cls(num_pairs, classifier_input, batch_size) )
        predictions['combined']['cr'] = self.cr_cls(num_pairs, classifier_input, batch_size)
        
        predictions['combined']['mr'] = self.mr_cls(num_pairs, classifier_input, 
                                                    batch_size)
        predictions['combined']['adj_mat'] = pred_adj_mat


        return predictions


    def _load_link_fun(self, config):

        if not os.path.exists(config['model_saving_path']):
            os.makedirs(config['model_saving_path'])

        best_model_file = os.path.join(
            config['model_saving_path'], os.pardir, 'graph', 'model_best.pth')

        if os.path.isfile(best_model_file):
            checkpoint = torch.load(best_model_file)
            self.link_fun.load_state_dict(checkpoint['state_dict'])