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