import os
from utils.utils import read_data
from torch.utils.data import Dataset
import torch
import pickle as pkl

class vsgnet_dataset(Dataset):
    """Class for ObjectObjectInteraction Dataset"""

    def __init__(self, config, split):
        """
        Args:
            dataset_config_file: Path to the config file with various configs.
                                 Has information about which features to use
        """
        self.config = config
        self.split = split
        self.filter_for_split()
        self.dataset_folder = self.config['dataset_folder']

        self.dataset = read_data(self.dataset_location)
        
        if self.config['overfit'] and split=='train':
            self.split_file_list = self.split_file_list[:self.config['train_batch_size']]
            # self.dataset = self.dataset[:1]

    def filter_for_split(self):
        
        split_file_dict = pkl.load(open(self.config['split_file_loc'], 'rb'))
        
        self.split_file_list = []
        for k in list(split_file_dict.keys()):
            if split_file_dict[k] == self.split:
                self.split_file_list.append(k + '_5.pt')
            
    def read_file_idx(self, file_idx):

        file_loc = os.path.join(self.dataset_folder, self.split_file_list[file_idx])
        data = torch.load(file_loc)
        return data

    def __len__(self):
        return len(self.split_file_list)

    def __getitem__(self, idx):
        
        data_item = self.read_file_idx(idx)

        return data_item, idx