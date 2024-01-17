
from torch.utils.data import Dataset
from tqdm import tqdm
import tifffile
import numpy as np
import random
import torch
import os
import cv2
import math
import yaml

#NOTE: Sometimes it's cleaner to just make two or three classes. For example ValExampleDataset
#NOTE: I would say it depends on how much they need to differ, if it's just a couple of lines if/else works fine.
class ExampleDataset(Dataset):

    def __init__(self, config, part = 'train'):
        self.config = config
        self.part = part
        self.transform = config.transform
        self.base_path_data = config.base_path_data
        self.validation_size = config.validation_size

        self.data = []
        self.labels = []
        
        print('Constructing '+self.part+' set...')

        #path to input
        INPUT_BASE_PATH = os.path.join(self.base_path_data, 'flair_2_toy_aerial_' + 'train') #or self.part if there is val
        #path to labels
        LABEL_BASE_PATH = os.path.join(self.base_path_data, 'flair_2_toy_labels_' + 'train')

        #here we should also save stds and norms of things

        #can write nicer code for this, not two loops, but I'll wait for the real data
        input_tif_paths = [] #vector of all aearial photos
        for root, dirs, files in os.walk(INPUT_BASE_PATH):
            for file in files:
                if file.endswith(".tif"):
                    input_tif_paths.append(os.path.join(root, file))

        label_tif_paths = []
        for root, dirs, files in os.walk(LABEL_BASE_PATH):
            for file in files:
                if file.endswith(".tif"):
                    label_tif_paths.append(os.path.join(root, file))

        #where to split the training data into training and validation
        split_point = math.floor(len(input_tif_paths)*(1-self.validation_size))
        
        if self.part == 'train':
            print(split_point)
            input_tif_paths = input_tif_paths[0:split_point]
            label_tif_paths = label_tif_paths[0:split_point]

        elif self.part == 'val':
            print(len(input_tif_paths)-split_point)
            input_tif_paths = input_tif_paths[split_point:]
            label_tif_paths = label_tif_paths[split_point:]
        
        # Assert that the number of aerial and sentinel tifs are the same
        assert len(input_tif_paths) == len(label_tif_paths)

        #read all input files
        for i, path in tqdm(enumerate(input_tif_paths)):
            img = np.array(tifffile.imread(path))
            if self.config.scale:
                img = self._rescale_to_power_of_two(img, False)
            self.data.append(img)

        for i, path in tqdm(enumerate(label_tif_paths)):
            label = np.array(tifffile.imread(path))
            if self.config.scale:
                label = self._rescale_to_power_of_two(label, True)
            self.labels.append(label)


        #print(self.data) #38 rows: 38 images, 512x512 pixels, 5 channels

    #called by data_set = ExampleDataset(config, part)
    #data_point = data_set[i], where i is the index of the desired object
    def __getitem__(self, index):
        #here we should also crop, and apply transforms
        img = self.data[index][:, :, :3] #take only RGB
        img = np.transpose(img, (2, 0, 1)) #transpose so channel before size
        label = self.data[index][:, :, :3] #take only RGB
        label = np.transpose(img, (2, 0, 1)) 
        return torch.tensor(img, dtype = torch.float), torch.tensor(label, dtype = torch.long)
    
    def __len__(self):
        # Return the number of samples in the dataset
        return len(self.data)

    def _rescale_to_power_of_two(data, is_label):
        #good to resize since computers likes powers of two
        h = data.shape[0]
        w = data.shape[1]
        power_h = math.ceil(np.log2(h) / np.log2(2))
        power_w = math.ceil(np.log2(w) / np.log2(2))
        if 2**power_h != h or 2**power_w != w:
            if not is_label: #i.e. is normal image, guess it just depends on nbr channels
                data = cv2.resize(data, dsize=(2**power_h, 2**power_w), interpolation=cv2.INTER_CUBIC)
            else: 
                data = cv2.resize(data, dsize=(2**power_h, 2**power_w), interpolation=cv2.INTER_NEAREST)
        return data


#NOTE: Given the from of loop these functions should be in any dataset module that you design (given that you keep it unchanged)
def train_set(config):
    return ExampleDataset(config)

def val_set(config):
    return ExampleDataset(config, part = 'val')

def test_set(config):
    pass

#code to load the configs
def load_config(file_path): 
    with open(file_path, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
            return config
        except yaml.YAMLError as e:
            print(f"Error loading YAML file: {e}")
            return None
             
class DotDict(dict):
    """Custom dictionary class to provide dot notation access."""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


#config_path = '../config/dataset/read_set.yaml'
#config_data = load_config(config_path)
#config = DotDict(config_data)

#train_set = train_set(config)
#val_set = val_set(config)



