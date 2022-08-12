import os
from utils.utils import read_data
from torch.utils.data import Dataset

class dataset(Dataset):
    """Class for ObjectObjectInteraction Dataset"""

    def __init__(self, config, split):
        """
        Args:
            dataset_config_file: Path to the config file with various configs.
                                 Has information about which features to use
        """
        self.config = config

        self.dataset_location = os.path.join(
                                             self.config['dataset_root'], 
                                            )

        self.dataset = read_data(self.dataset_location)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        
        return self.dataset[idx]
        # return self.dataset[idx], idx