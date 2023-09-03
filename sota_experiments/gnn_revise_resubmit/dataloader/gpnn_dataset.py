import os
from utils.utils import read_data
from torch.utils.data import Dataset
import torch
from torch_geometric import utils as tg_utils

class gpnn_dataset(Dataset):
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
        # self.dataset = self.full_dataset
        self.select_split()
        
        if self.config['overfit'] and split=='train':
            self.dataset = self.dataset[:self.config['train_batch_size']]

    def select_split(self):
        
        self.dataset = []

        for d in self.full_dataset:
            
            yt_id = d['metadata']['yt_id']
            frame_no = d['metadata']['frame no.']
            
            temp_key = yt_id + '_' + frame_no
            
            temp_split = self.split_dict[temp_key]
            if temp_split == self.split:
                self.dataset.append(d)
        print("FLAG DATASET", len(self.dataset))
        del self.full_dataset

    def __len__(self):
        #return # self.config['train_batch_size']
        return len(self.dataset)

    def __getitem__(self, idx):
        data_item = self.dataset[idx]
        return data_item, idx

