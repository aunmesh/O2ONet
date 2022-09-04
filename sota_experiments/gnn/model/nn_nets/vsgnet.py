import torch
import torch.nn as nn
from utils.utils import aggregate

from nn_modules.visual_branch_vsgnet import VisualBranch_vsgnet
from nn_modules.spatial_branch_vsgnet import SpatialBranch_vsgnet
from nn_modules.graphical_branch_vsgnet import GraphicalBranch_vsgnet

class vsgnet(nn.Module):
    '''
    vsgnet stands for Visual Spatial Graphical Network

    Uses:
    '''

    def __init__(self, config):

        '''
        Constructor for vsgnet
        Args:
        vsgnet_config_file : Has the various configurations necessary for this network
                              This config file has information on the static and temporal feature
                              dimensions along with other things.
        '''

        super(vsg_net, self).__init__()
        self.config = config

        # creating the gcn
        self.graphical_branch = GraphicalBranch_vsgnet(config)
        self.spatial_branch = SpatialBranch_vsgnet(config)
        self.visual_branch = VisualBranch_vsgnet(config)
        
        self.refined_branch_classifiers = {}
        self.graphical_branch_classifiers = {}
        self.spatial_branch_classifiers = {}
        self.relation_keys = ['lr', 'mr', 'cr']

        for k in self.relation_keys:
            temp_dimension = 'refined_branch_' + k + '_classifier_dimension'
            self.refined_branch_classifiers[k] = self.make_classifier(temp_dimension)
            
            temp_dimension = 'spatial_branch_' + k + '_classifier_dimension'
            self.spatial_branch_classifiers[k] = self.make_classifier(temp_dimension)

            temp_dimension = 'graphical_branch_' + k + '_classifier_dimension'
            self.graphical_branch_classifiers[k] = self.make_classifier(temp_dimension)
    

    def make_classifier(self, dimensions, key):

        classifier_layers = nn.ModuleList()  # for storing all the transform layers

        for i in range(len(dimensions) - 1):
            curr_d = dimensions[i]
            next_d = dimensions[i+1]

            temp_fc_layer = nn.Linear(curr_d, next_d)
            classifier_layers.append(temp_fc_layer)
            
            if i < len(dimensions) - 2:
                classifier_layers.append(nn.ReLU())
        
        if key == 'scr':
            classifier_layers.append( nn.Softmax(dim=1) )
        
        else:
            classifier_layers.append( nn.Sigmoid(dim=1) )
            

        return classifier_layers


    def pair_graphical_branch_output(
                                            self, graphical_branch_output, num_rels, 
                                            num_obj, obj_pairs
                                          ):

        graphical_branch_output_dim = graphical_branch_output.shape[-1]

        tot_num_rels = torch.sum(num_rels)
        graphical_branch_paired = torch.zeros(
                                                    (
                                                    tot_num_rels, 
                                                    graphical_branch_output_dim
                                                    ), 
                                                    device = self.config['device'], 
                                                    dtype=torch.double
                                                   )
        
        batch_size = num_rels.shape[0]
                
        # verify this again carefully
        for curr_batch in range(batch_size):
            
            curr_num_rels = num_rels[curr_batch]
            object_index_offset = torch.sum(num_obj[:curr_batch])
            relation_index_offset = torch.sum(num_rels[:curr_batch])
            
            for j in range(curr_num_rels):

                obj_ind_0, obj_ind_1 = obj_pairs[curr_batch, j]
                
                obj_ind_0 += object_index_offset
                obj_ind_1 += object_index_offset

                obj_vec_0 = graphical_branch_output[obj_ind_0]
                obj_vec_1 = graphical_branch_output[obj_ind_1]

                temp_obj_vector = aggregate(obj_vec_0, obj_vec_1, 'mean')
                
                temp_relation_index = relation_index_offset + j
                
                graphical_branch_paired[temp_relation_index, :] = temp_obj_vector
        
        return graphical_branch_paired


    def forward(self, data_item):
        '''
        Input: data_item

        Output: dictionary with the predictions for all the classes
        '''

       
        
        object_branch_output, f_oo_vis  = self.visual_branch(data_item)
        spatial_branch_output = self.spatial_branch(data_item)
        graphical_branch_output = self.graphical_branch(data_item, object_branch_output)
        
        # refine the output from spatial_branch and f_oo_vis
        spatial_visual_refined_features = f_oo_vis * spatial_branch_output
        
        # all the incoming features are ordered according to object_pairs for each batch elements
        
        # classify using the spatial branch only

        res_spatial = {}
        for k in self.relation_keys:
            res_spatial[k] = self.spatial_branch_classifiers[k](spatial_branch_output)

        # classify using the refined features
        res_refined = {}
        for k in self.relation_keys:
            res_refined[k] = self.refined_branch_classifiers[k](spatial_visual_refined_features)
        
        # classify using the graphical features

        num_obj = data_item['num_obj']
        obj_pairs = data_item['object_pairs']
        num_rels = data_item['num_relation']
        graphical_branch_output_paired = self.pair_graphical_branch_output(
                                                            graphical_branch_output,
                                                            num_rels, num_obj, obj_pairs
                                                                           )
        res_graphical = {}
        for k in self.relation_keys:
            res_graphical[k] = self.graphical_branch_classifiers[k](
                                            graphical_branch_output_paired
                                                                    )
        
        result = {}
        for k in self.relation_keys:
            result[k] = res_spatial[k] * res_refined[k] * res_graphical[k]
        
        return result