import torch
import torch.nn as nn
from utils.utils import aggregate


# 3 branches - Visual Branch, Spatial Branch, Graphical Branch

# Visual Branch 
# 2 subbranches = object branch and context branch
# Object branch input = resnet 52 feature map, object bounding box. Output = feature vector for object
# Context branch input = resnet, output = feature vector for context
# visual branch combines them together and outputs the feature vector for the 2 objects together




class vsg_net(nn.Module):
    '''
    vsg_net stands for Visual Spatial Graphical Network

    Uses:
    '''

    def __init__(self, config):

        '''
        Constructor for vsgnet
        Args:
        vsg_net_config_file : Has the various configurations necessary for this network
                              This config file has information on the static and temporal feature
                              dimensions along with other things.
        '''

        super(vsg_net, self).__init__()
        self.config = config

        # creating the gcn
        self.device = config['device']
        


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

    def forward(self, data_item):
        '''
        Input:

        Output:
        '''

        # Flow

        frame_feature_map = data_item['frame_deep_features']
        bboxes = data_item['bboxes']
        num_obj = data_item['num_obj']
        obj_pairs = data_item['object_pairs']
        num_rels = data_item['num_relation']
        
        visual_branch_output = self.visual_branch(frame_feature_map, bboxes, num_obj, obj_pairs, num_rels)
        
        spatial_branch_output = self.spatial_branch(frame_feature_map, bboxes, num_obj, obj_pairs, num_rels)
        
        
        
        
        
        return predictions