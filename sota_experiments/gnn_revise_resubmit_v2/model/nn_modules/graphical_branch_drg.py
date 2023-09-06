import torch
import torch.nn as nn
from model.nn_modules.gnn import GNN
from utils.utils import aggregate

class GraphicalBranch_drg(torch.nn.Module):
    
    def __init__(self, config):
        
        super(GraphicalBranch_drg, self).__init__()
        
        self.config = config
        self.edge_indices = {}

        for num in range(30):
            temp_edge_index = self.generate_fc_edge_index(num)
            self.edge_indices[num] = temp_edge_index
        
        self.graph_dimensions = self.config['gnn_dimensions']
        self.gnn = GNN(config)
    
    def generate_fc_edge_index(self, num_obj):
        '''
        generates edge index for a fully connected graph
        '''

        num_edges = num_obj**2
        temp_edge_index = torch.zeros(
                                        (2, num_edges), 
                                        dtype=torch.long,
                                        device=self.config['device']
                                        )
        curr_ind = 0
        
        for t1 in range(num_obj):
            for t2 in range(num_obj):

                temp_edge_index[0, curr_ind] = t1
                temp_edge_index[1, curr_ind] = t2
                curr_ind+=1

        return temp_edge_index


    def filter_tensor_for_classifier( self, graphical_branch_output, 
                        slicing_tensor, object_pairs, 
                        num_rel):
        
        total_rels = int(torch.sum(num_rel))
        dim_fmap = graphical_branch_output.shape[-1]
        filtered_tensors = torch.zeros( (total_rels, dim_fmap),
                                      device = self.config['device'])
        
        batch_size = object_pairs.shape[0]
        curr_ind = 0
        
        for b in range(batch_size):
            temp_num_rel = int(num_rel[b])
        
            for i in range(temp_num_rel):
                obj_ind_0, obj_ind_1 = object_pairs[b, i]
                obj_ind_min = min( int(obj_ind_0), int(obj_ind_1))
                obj_ind_max = max( int(obj_ind_0), int(obj_ind_1))
                # comparison_tensor = torch.tensor([b, obj_ind_min, obj_ind_max],
                                                #  device = self.config['device'])
                # rel_index = torch.where(slicing_tensor == comparison_tensor)
                
                temp_0 = (slicing_tensor[:,0] == b)
                temp_1 = (slicing_tensor[:,1] == obj_ind_min)
                temp_2 = (slicing_tensor[:,2] == obj_ind_max)
                
                temp = temp_0 * temp_1 * temp_2
                temp_loc = torch.where(temp)[0]
                req_ind = int(temp_loc)
                
                filtered_tensors[curr_ind, :] = graphical_branch_output[req_ind, :]
                curr_ind+=1
        
        return filtered_tensors


    def forward(self, data_item, spatial_branch_feature_map, slicing_tensor):
        '''
        spatial_branch_feature_map has features for every combination
        slicing tensor is a tensor with 3 columns. 
        1 for batch_index, 1 for i, 1 for j

        Edge index corresponding to a fully connected graph
        make the edge index for the entire batch.
        pass it to the GNN.
        Get the outputs.
        Extract the relevant output pairs in a separate tensor.
        Then return that tensor.
        '''
        num_obj = data_item['num_obj']
        batch_size = num_obj.shape[0]
        
        # creating edge index for the current batch
        edge_indices = []
        offset=0
        
        for b in range(batch_size):
            n = num_obj[b]
            nc2 = int(n*(n-1)/2)
            temp_edge_index = self.edge_indices[nc2].clone() + offset
            edge_indices.append(temp_edge_index)
            offset+=nc2

        combined_edge_index = torch.cat(edge_indices, dim=1)

        # Perform graphical convolution using pytorch geometric
        graphical_obj_features = self.gnn(
                                            spatial_branch_feature_map, 
                                            combined_edge_index
                                         )
        
        obj_pairs = data_item['object_pairs']
        num_rel = data_item['num_relation']
        filtered_tensor = self.filter_tensor_for_classifier(graphical_obj_features,
                                                            slicing_tensor, obj_pairs,
                                                            num_rel)

        return filtered_tensor