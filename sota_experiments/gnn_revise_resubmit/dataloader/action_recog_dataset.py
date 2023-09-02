import os
from utils.utils import read_data
from torch.utils.data import Dataset
import torch
from torch_geometric import utils as tg_utils


class action_recog_dataset(Dataset):
    """Class for ObjectObjectInteraction Dataset"""


    def __init__(self, config, split):
        """
        Args:
            dataset_config_file: Path to the config file with various configs.
                                 Has information about which features to use
        """
        self.config = config
        self.split = split
        
        self.dataset_root = self.config['dataset_root']
        
        self.classes = [
                            'AssembleCabinet', 'ChangeCarTire', 'FuelCar',
                             'InstallBicycleRack', 'InstallShowerHead', 'ParkParallel',
                             'PolishCar', 'ReplaceBatteryOnTVControl', 'ReplaceDoorKnob',
                             'ReplaceToiletSeat', 'UseJack'
                             ]
        self.class_index = {}
        
        for i, c in enumerate(self.classes):
            self.class_index[c] = i
        
        self.keys_to_send_to_gpu = ['i3d_fmap']
        
        self.get_data()

    #         
    def get_data(self):
        
        from glob import glob
        
        glob_str = os.path.join(self.dataset_root, '*.pt')
        temp_file_list = glob(glob_str)
        
        f = open(self.config['split_file'], 'rb')
        import pickle
        self.split_dict = pickle.load(f)    # Dictionary which stores the split information for each file
        f.close()
        
        self.file_list = []
        # Loading all the files which belong to the split
        for f in temp_file_list:
            
            fname = f.split('/')[-1]
            fname = fname.split('.')[0][:-2]    # filtering the string according to the names of the keys in the split dictionary
            
            temp_split = self.split_dict[fname]
            
            if temp_split == self.split:
                self.file_list.append(f)
        
        
        self.data = []
        
        for f in self.file_list:
            temp_data = torch.load(f)
            temp_class_name = temp_data['metadata']['activity name']
            class_index = self.class_index[temp_class_name]
            temp_data['action_index'] = class_index #.to( self.config['device'] )
            
            for k in self.keys_to_send_to_gpu:
                temp_data[k] = temp_data[k].to(self.config['device'])

            self.data.append(temp_data)

        print("Length of" , self.split, "dataset is", len(self.data))
            
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        # data = torch.load( self.file_list[idx] )
        # class_index = self.class_index[ data['metadata']['activity name'] ]
        # data['action_index'] = class_index #.to( self.config['device'] )
        
        # for k in self.keys_to_send_to_gpu:
        #     data[k] = data[k].to(self.config['device'])
        
        # return data, idx
        return self.data[idx], idx