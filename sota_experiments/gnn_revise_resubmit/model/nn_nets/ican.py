import os

import torch
import torch.nn as nn
import torch.autograd

from model.nn_modules.spatial_branch_ican import SpatialBranch_ican
from model.nn_modules.visual_branch_ican import VisualBranch_ican


class iCAN(torch.nn.Module):
    """
    ican_multiplication has a naive way of combining the 2 streams.
    It classifies independently using both the streams and then multiplies them together to get
    the final classifier. While 
    """
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
        for k in ['lr', 'mr']:
            res_combined[k] = res_visual[k] * res_spatial[k]
        
        res_combined['cr'] = res_visual['cr'] * res_spatial['cr']
        norm = torch.sum(res_combined['cr'],dim=1).unsqueeze(1)
        res_combined['cr']/=norm
        
        res_all_stream = {}
        res_all_stream['visual'] = res_visual
        res_all_stream['spatial'] = res_spatial
        res_all_stream['combined'] = res_combined

        return res_all_stream

class iCAN_v2(torch.nn.Module):

    def __init__(self, config):

        super(iCAN, self).__init__()
        self.config = config.copy()
        
        self.visual_branch = VisualBranch_ican(self.config)
        self.spatial_branch = SpatialBranch_ican(self.config)
        
        #######################
        
        self.classifiers = {}
        self.relation_keys = ['lr', 'mr', 'cr']

        for k in self.relation_keys:

            temp_dimension = self.config['combined_branch_' + k + '_classifier_dimension']
            self.classifiers[k] = self.make_classifier(temp_dimension, k)

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

        concatenated_output = torch.cat([spatial_branch_output, visual_branch_output], dim=-1)
        
        # classify using the features

        res_combined = {}
        for k in self.relation_keys:
            res_combined[k] = self.classifiers[k](concatenated_output)

        res_all_stream = {}
        res_all_stream['combined'] = res_combined

        return res_all_stream



class iCAN_v1(torch.nn.Module):

    def __init__(self, config):

        super(iCAN, self).__init__()
        self.config = config.copy()
        
        self.visual_branch = VisualBranch_ican(self.config)
        self.spatial_branch = SpatialBranch_ican(self.config)
        
        
        self.spatial_branch_transform_elements = []
        self.spatial_branch_transform_elements.append( nn.Linear(512, 512, device=self.config['device']) )
        self.spatial_branch_transform_elements.append( nn.ReLU().to(self.config['device']) )
        
        self.spatial_branch_transform = nn.Sequential(*self.spatial_branch_transform_elements)
        
        self.visual_branch_transform_elements = []
        self.visual_branch_transform_elements.append( nn.Linear(1024, 512, device=self.config['device']) )
        self.visual_branch_transform_elements.append( nn.ReLU().to(self.config['device']) )
        
        self.visual_branch_transform = nn.Sequential(*self.visual_branch_transform_elements)
        
        
        self.concatenated_transform_elements = []
        self.concatenated_transform_elements.append(nn.Linear(1024, 512, device=self.config['device']))
        self.concatenated_transform_elements.append(nn.ReLU().to(self.config['device']))
        
        self.concatenated_transform = nn.Sequential(*self.concatenated_transform_elements)



        #######################
        
        self.classifiers = {}
        self.relation_keys = ['lr', 'mr', 'cr']

        for k in self.relation_keys:

            temp_dimension = self.config['combined_branch_' + k + '_classifier_dimension']
            self.classifiers[k] = self.make_classifier(temp_dimension, k)

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
        spatial_branch_output = self.spatial_branch_transform(spatial_branch_output)
        
        visual_branch_output = self.visual_branch( data_item )
        visual_branch_output = self.visual_branch_transform(visual_branch_output)

        concatenated_output = torch.cat([spatial_branch_output, visual_branch_output], dim=-1)
        concatenated_output = self.concatenated_transform(concatenated_output)
        
        # classify using the features

        res_combined = {}
        for k in self.relation_keys:
            res_combined[k] = self.classifiers[k](concatenated_output)

        res_all_stream = {}
        res_all_stream['combined'] = res_combined

        return res_all_stream






