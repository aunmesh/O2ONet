import os
from utils.utils import read_data
from torch.utils.data import Dataset
import torch
from torch_geometric import utils as tg_utils

class dataset(Dataset):
    """Class for ObjectObjectInteraction Dataset"""

    def __init__(self, config, split):
        """
        Args:
            dataset_config_file: Path to the config file with various configs.
                                 Has information about which features to use
        """

        self.config = config
        self.split = split
        
        self.full_dataset = read_data(self.config['full_dataset_location'])

        self.split_dict = read_data(config['split_dict_location'])
        
        self.select_split()
        
        # if self.config['overfit'] and split=='train':
        #     self.dataset = self.dataset[:self.config['train_batch_size']]

    def select_split(self):
        
        self.dataset = []

        for d in self.full_dataset:
            
            yt_id = d['metadata']['yt_id']
            frame_no = d['metadata']['frame no.']
            
            temp_key = yt_id + '_' + frame_no
            
            temp_split = self.split_dict[temp_key]
            if temp_split == self.split:
                self.dataset.append(d)

        del self.full_dataset

    def __len__(self):

        return len(self.dataset)
        # return len(self.indices)

    def __getitem__(self, idx):
        
        if self.config['model_name'] == 'hgat' or self.config['model_name'] == 'GraphTransformer':
            data_item = self.modify_dataitem(idx)
            return data_item, idx

        else:            
            data_item = self.dataset[idx]
            return data_item, idx



    def modify_dataitem(self, idx):
        '''
        Construct Edge Indices
        '''

        data_item = self.dataset[idx]
        
        # Get the threshold 
        t_iou = self.config['iou_threshold']
        t_dis = self.config['dis_threshold']
        
        # Edge Index has to be made using the central frame only
        # Thus need the matrices of iou and distance for the central frame
        central_frame_index = int(self.config['gif_size']/2)
        temp_num_obj = data_item['num_obj']
        
        temp_iou = data_item['iou'][:temp_num_obj, :temp_num_obj, central_frame_index]
        temp_dis = data_item['distance'][:temp_num_obj, :temp_num_obj, central_frame_index]
        
        # Matrix to be used for deciding edges between the nodes
        temp_mat = (temp_iou > t_iou) + (temp_dis < t_dis)
        temp_index = torch.where(temp_mat)
        temp_length = temp_index[0].shape[0]
        
        temp_edge_index = torch.ones((2, temp_length), dtype=torch.long)
        temp_edge_index[0,:] = temp_index[0]
        temp_edge_index[1,:] = temp_index[1]
        
        # remove self loops and make it undirected
        temp_edge_index = tg_utils.remove_self_loops(temp_edge_index)[0]
        temp_edge_index = tg_utils.add_self_loops(temp_edge_index)[0]
        temp_edge_index = tg_utils.to_undirected(temp_edge_index)

        max_num_edges = int(self.config['max_num_edges'])
        edge_index = -1 * torch.ones((2, max_num_edges), dtype=torch.long, device=self.config['device'])

        num_edges = temp_edge_index.shape[1]
        data_item['num_edges'] = num_edges

        edge_index[:,:num_edges] = temp_edge_index        
        data_item['edge_index'] = edge_index
        
        return data_item





    def modify_dataitem_2(self, idx):
        '''
        Construct Edge Indices
        '''

        data_item = self.dataset[idx]
        
        # Get the threshold 
        t_iou = self.config['iou_threshold']
        t_dis = self.config['dis_threshold']
        
        # Edge Index has to be made using the central frame only
        # Thus need the matrices of iou and distance for the central frame
        central_frame_index = int(self.config['gif_size']/2)
        temp_num_obj = data_item['num_obj']
        
        temp_iou = data_item['iou'][:temp_num_obj, :temp_num_obj, central_frame_index]
        temp_dis = data_item['distance'][:temp_num_obj, :temp_num_obj, central_frame_index]
        
        # Matrix to be used for deciding edges between the nodes
        temp_mat = (temp_iou > t_iou) + (temp_dis < t_dis)
        temp_index = torch.where(temp_mat)
        temp_length = temp_index[0].shape[0]
        
        temp_edge_index = torch.ones((2, temp_length), dtype=torch.long)
        temp_edge_index[0,:] = temp_index[0]
        temp_edge_index[1,:] = temp_index[1]
        
        # remove self loops and make it undirected
        temp_edge_index = tg_utils.remove_self_loops(temp_edge_index)[0]
        temp_edge_index = tg_utils.add_self_loops(temp_edge_index)[0]
        temp_edge_index = tg_utils.to_undirected(temp_edge_index)

        max_num_edges = int(self.config['max_num_edges'])
        edge_index = -1 * torch.ones((2, max_num_edges), dtype=torch.long, device=self.config['device'])

        num_edges = temp_edge_index.shape[1]
        data_item['num_edges'] = num_edges

        edge_index[:,:num_edges] = temp_edge_index        
        data_item['edge_index'] = edge_index
        
        return data_item

