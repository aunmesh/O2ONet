import os

import torch
import torch.nn as nn
import torch.autograd

from model.nn_modules.spatial_branch_ican import SpatialBranch_ican
from model.nn_modules.visual_branch_ican import VisualBranch_ican

class iCAN(torch.nn.Module):

    def __init__(self, config):

        super(iCAN, self).__init__()
        self.config = config.copy()
        
        self.visual_branch = VisualBranch_ican(self.config)
        self.spatial_branch = SpatialBranch_ican(self.config)
        
        self.spatial_branch_classifiers = {}
        self.visual_branch_classifiers = {}
        
        self.relation_keys = ['lr', 'mr', 'cr']

        for k in self.relation_keys:

            temp_dimension = self.config['spatial_branch_' + k + '_classifier_dimension']
            self.spatial_branch_classifiers[k] = self.make_classifier(temp_dimension, k)
            
            temp_dimension = self.config['visual_branch_' + k + '_classifier_dimension']
            self.visual_branch_classifiers[k] = self.make_classifier(temp_dimension, k)

    def make_classifier(self, dimensions, key):

        classifier_layers = []  # for storing all the transform layers

        for i in range(len(dimensions) - 1):
            curr_d = dimensions[i]
            next_d = dimensions[i+1]
            temp_fc_layer = nn.Linear(curr_d, next_d)
            classifier_layers.append(temp_fc_layer)
            
            if i < len(dimensions) - 2:
                classifier_layers.append(nn.ReLU())
        
        if key == 'cr':
            classifier_layers.append( nn.Softmax(dim=1) )
        
        else:
            classifier_layers.append( nn.Sigmoid() )
        
        model = nn.Sequential(*classifier_layers).to(self.config['device'])

        return model


    def forward(self, data_item):

        spatial_branch_output = self.spatial_branch(data_item)
        visual_branch_output = self.visual_branch( data_item )
        
        # classify using the features

        res_visual = {}
        for k in self.relation_keys:
            res_visual[k] = self.visual_branch_classifiers[k](visual_branch_output)

        res_spatial = {}
        for k in self.relation_keys:
            res_spatial[k] = self.spatial_branch_classifiers[k](spatial_branch_output)

        
        res_combined = {}
        for k in self.relation_keys:
            res_combined[k] = res_visual[k] * res_spatial[k]
        
        res_all_stream = {}
        res_all_stream['visual'] = res_visual
        res_all_stream['spatial'] = res_spatial
        res_all_stream['combined'] = res_combined
        
        return res_all_stream