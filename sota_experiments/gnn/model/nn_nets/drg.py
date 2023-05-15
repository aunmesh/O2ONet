import os

import torch
import torch.nn as nn
import torch.autograd

from model.nn_modules.graphical_branch_drg import GraphicalBranch_drg
from model.nn_modules.spatial_branch_drg import SpatialBranch_drg
from model.nn_modules.object_branch_drg import ObjectBranch_drg

class DRG(torch.nn.Module):

    def __init__(self, config):

        super(DRG, self).__init__()
        self.config = config.copy()
        
        self.graphical_branch = GraphicalBranch_drg(self.config)
        # self.object_branch = ObjectBranch_drg(self.config)
        self.spatial_branch = SpatialBranch_drg(self.config)
        
        self.graphical_branch_classifiers = {}

        self.relation_keys = ['lr', 'mr', 'cr']

        for k in self.relation_keys:

            temp_dimension = self.config['graphical_branch_' + k + '_classifier_dimension']
            self.graphical_branch_classifiers[k] = self.make_classifier(temp_dimension, k)
            
            # temp_dimension = self.config['object_branch_' + k + '_classifier_dimension']
            # self.object_branch_classifiers[k] = self.make_classifier(temp_dimension, k)



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
            # classifier_layers.append( nn.Softmax(dim=1) )
            classifier_layers.append( nn.LogSoftmax(dim=1) )
        
        # else:
        #     classifier_layers.append( nn.Sigmoid() )
        model = nn.Sequential(*classifier_layers).to(self.config['device'])

        return model


    def forward(self, data_item):

        # object_branch_output_paired = self.object_branch(data_item)
        spatial_feature_map, spatial_slicing_tensor = self.spatial_branch(data_item)
        graphical_branch_output_paired = self.graphical_branch(
                                                                data_item, 
                                                                spatial_feature_map,
                                                                spatial_slicing_tensor
                                                               )

        # classify using the features
        res_graphical = {}
        for k in self.relation_keys:
            res_graphical[k] = self.graphical_branch_classifiers[k](
                                            graphical_branch_output_paired
                                                                    )

        # res_object = {}
        # for k in self.relation_keys:
        #     res_object[k] = self.object_branch_classifiers[k](object_branch_output_paired)

        
        # res_combined = {}
        # for k in self.relation_keys:
        #     res_combined[k] = res_object[k] * res_graphical[k]

        # res_combined = {}
        # for k in self.relation_keys:
        #     res_combined[k] = res_graphical[k]

        res_combined = res_graphical
                
        res_all_stream = {}
        # res_all_stream['graphical'] = res_graphical
        # res_all_stream['object'] = res_object
        res_all_stream['combined'] = res_combined
        
        return res_all_stream