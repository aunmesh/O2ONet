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
        
        self.dataset = read_data(self.config['full_dataset_location'])
                
        self.indices = list(range(len(self.dataset)))

        # if self.config['overfit'] and split=='train':
        #     self.dataset = self.dataset[:self.config['train_batch_size']]


    def __len__(self):
        #return # self.config['train_batch_size']
        # return len(self.dataset)
        return len(self.indices)

    def set_indices(self, new_indices):
        self.indices = new_indices

    def __getitem__(self, idx):
        
        actual_idx = self.indices[idx]
        
        if self.config['model_name'] == 'hgat' or self.config['model_name'] == 'GraphTransformer':
            
            data_item = self.modify_dataitem(actual_idx)
            return data_item, actual_idx

        else:            
            data_item = self.dataset[actual_idx]
            return data_item, idx



    def modify_dataitem(self, idx):
        '''
        Construct Edge Indices
        '''
        
        ablated_feat = self.config['ablated_feat']
        
        

        return data_item

