import torch
import torch.nn as nn
from model.nn_modules.gnn import GNN
from utils.utils import aggregate

class GraphicalBranch_vsgnet(torch.nn.Module):
    
    def __init__(self, config):
        
        super(GraphicalBranch_vsgnet, self).__init__()
        
        self.config = config
        self.edge_indices = {}

        for num in range(12):
            temp_edge_index = self.generate_fc_edge_index(num)
            self.edge_indices[num] = temp_edge_index
        
        self.graph_dimensions = self.config['gnn_dimensions']
        self.gnn = GNN(config)
        
    
    def generate_fc_edge_index(self, num_obj):
        '''
        generates edge index for a fully connected graph
        '''

        num_edges = num_obj**2
        temp_edge_index = -1 * torch.ones(
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
            
    def forward(self, data_item, object_branch_output):
        
        # Generate a fully connected graph
        # 1. generate the slicing dictionary for the nodes.
        # 2. generate the edges for the fully connected graph. Edge index is of shape [2, num_edges]
        # 3. send it for convolution.
        # 4. get the node features.


        num_obj = data_item['num_obj']
        
        batch_size = num_obj.shape[0]
        obj_ind_offset = 0        

        edge_indices = []

        edge_slicing = []
        node_slicing = []

        for curr_batch in range(batch_size):
            
            temp_num_obj = int(num_obj[curr_batch])
            temp_edge_index = self.edge_indices[temp_num_obj].clone() + obj_ind_offset
            edge_indices.append(temp_edge_index)
            
            temp_slice = [curr_batch] * temp_num_obj
            node_slicing = node_slicing + temp_slice
            
            temp_edge_slicing = [curr_batch] * (temp_num_obj ** 2)
            edge_slicing = edge_slicing + temp_edge_slicing

            obj_ind_offset = obj_ind_offset + temp_num_obj

        slicing_dict = {'node': node_slicing, 'edge': edge_slicing}
        combined_edge_index = torch.cat(edge_indices, dim=1)

        # Perform graphical convolution using pytorch geometric
        graphical_obj_features = self.gnn(object_branch_output, combined_edge_index)

        # output is the convolved embedding of each node
        return graphical_obj_features, slicing_dict