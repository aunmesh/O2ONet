import torch
import os
from utils.utils import read_data
from torch.utils.data import Dataset

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
        
        self.generate_file_list()

        if self.config['overfit'] and split=='train':
            self.dataset = self.dataset[:self.config['train_batch_size']]

    def generate_file_list(self):
        
        # 1. Get split dict
        # 2. Filter split dict to get the files
        # 3. Further filter the obtained files according to files which are present in the dataset location
        # 4. Generate full paths
        self.split_dict = read_data(self.config['split_dict_location'])
        
        candidates = [ k+'_5.pt' for k, v in self.split_dict.items() if v==self.split ]
        from glob import glob as glob

        all_feature_files = glob( os.path.join(self.config['dataset_location'], '*.pt') )
        self.file_list = []
        
        for c in candidates:
        
            temp_path = os.path.join(self.config['dataset_location'], c)
        
            if temp_path in all_feature_files:
                self.file_list.append(temp_path)
        
    def __len__(self):
        # return self.config[self.split + '_batch_size']
        return 32
        # return len(self.file_list)

    def __getitem__(self, idx):
        idx = 8
        data_item = torch.load( self.file_list[idx] )

        if data_item['i3d_feature_map'].shape[1:] != torch.Size([23, 40]):
            data_item['i3d_feature_map'] = torch.nn.functional.interpolate(
                                                        data_item['i3d_feature_map'].unsqueeze(0),
                                                        (23, 40)).squeeze()

        if data_item['2d_cnn_feature_map'].shape[1:] != torch.Size([23, 40]):
            data_item['2d_cnn_feature_map'] = torch.nn.functional.interpolate(
                                                        data_item['2d_cnn_feature_map'].unsqueeze(0), 
                                                        (23, 40)).squeeze()

        data_item['frame_deep_features'] = torch.cat([data_item['i3d_feature_map'], data_item['2d_cnn_feature_map']], dim=1)
        return data_item, idx




class vsgnet_dataset_old(Dataset):
    """Class for ObjectObjectInteraction Dataset"""

    def __init__(self, config, split):
        """
        Args:
            dataset_config_file: Path to the config file with various configs.
                                 Has information about which features to use
        """
        self.config = config
        self.split = split

        self.dataset_folder = self.config['dataset_root']
        split_annotation = self.config[split + '_annotations']

        self.dataset_location = os.path.join(
                                             self.config['dataset_root'], split_annotation
                                            )

        self.dataset = read_data(self.dataset_location)

        if self.config['overfit'] and split=='train':
            self.dataset = self.dataset[:self.config['train_batch_size']]



    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        
        data_item = self.dataset[idx]
        return data_item, idx