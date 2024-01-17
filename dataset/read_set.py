
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
        #self.transform = config.transform
        self.base_path_data = config.dataset.base_path_data
        self.validation_size = config.dataset.validation_size

        self.data = []
        self.labels = []
        
        print('Constructing '+self.part+' set...')

        #path to input
        INPUT_BASE_PATH = os.path.join(self.base_path_data, 'flair_2_toy_aerial_' + 'train') #or self.part if there is val
        #path to labels
        LABEL_BASE_PATH = os.path.join(self.base_path_data, 'flair_2_toy_labels_' + 'train')

        #here we should also save stds and norms of things

        input_tif_paths = self._read_paths(INPUT_BASE_PATH)
        label_tif_paths = self._read_paths(LABEL_BASE_PATH)

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

        #can write nicer code for this, not two loops, but I'll wait for the real data
        for i, path in tqdm(enumerate(input_tif_paths)):
            img = np.array(tifffile.imread(path))
            img = np.transpose(img, (2, 0, 1)) #transpose so channel before size
            if self.config.dataset.scale and not self.config.dataset.crop: #only rescaling to pow of two
                img = self._rescale_to_power_of_two(img, False)
                self.data.append(img)
            elif self.config.dataset.crop: #can handle both crop and rescale
                img_list = self._crop_images(img, False)
                self.data.extend(img_list)
            else:
                self.data.append(img)

        for i, path in tqdm(enumerate(label_tif_paths)):
            label = np.array(tifffile.imread(path))
            #label = np.transpose(label, (2, 0, 1)) have only one channel no need to transpose
            if self.config.dataset.scale and not self.config.dataset.crop: #only rescaling to pow of two
                label = self._rescale_to_power_of_two(label, True)
                self.labels.append(label)
            elif self.config.dataset.crop:
                label_list = self._crop_images(label, True)
                self.labels.extend(label_list)
            else:
                self.labels.append(label)

    #data_point = data_set[i], where i is the index of the desired object
    def __getitem__(self, index):
        #here we should also crop, and apply transforms
        img = self.data[index][:3, :, :] #take only RGBd
        label = self.data[index][:3, :, :] #take only RGB
    
        if self.part == 'val': #don't do transformations on validation data!
            return torch.tensor(img, dtype = torch.float), torch.tensor(label, dtype = torch.long)

        return torch.tensor(img, dtype = torch.float), torch.tensor(label, dtype = torch.long)
    
    def __len__(self):
        # Return the number of samples in the dataset
        return len(self.data)
    
    def _read_paths(self, BASE_PATH):
        tif_paths = []
        for root, dirs, files in os.walk(BASE_PATH):
            for file in files:
                if file.endswith(".tif"):
                    tif_paths.append(os.path.join(root, file))

        return tif_paths


    def _rescale_to_power_of_two(self, img, is_label):
        #good to resize since computers likes powers of two
        if is_label: #no channels
            h = img.shape[0]
            w = img.shape[1]
        else:
            h = img.shape[1]
            w = img.shape[2]
        power_h = math.ceil(np.log2(h) / np.log2(2))
        power_w = math.ceil(np.log2(w) / np.log2(2))
        if 2**power_h != h or 2**power_w != w:
            if not is_label: #i.e. is normal image, guess it just depends on nbr channels
                img = cv2.resize(img, dsize=(2**power_h, 2**power_w), interpolation=cv2.INTER_CUBIC)
            else: 
                img = cv2.resize(img, dsize=(2**power_h, 2**power_w), interpolation=cv2.INTER_NEAREST)
        return img
    
    def _crop_images(self, img, is_label):
        if is_label:
            img_h, img_w = img.shape[0], img.shape[1]
        else:
            img_h, img_w = img.shape[1], img.shape[2] #dims here depends on format of data
        crop_size = self.config.dataset.crop_size
        all_crops = []
        #do we want to have literally all possible crops, or do we want the disjoint ones? now disjoint
        #can be done in many different ways, otherways range(img_ - crop_size + 1, crop_size)
        for start_h in range(0, img_h - crop_size + 1, crop_size): 
            for start_w in range(0, img_w - crop_size + 1, crop_size):
                end_h = start_h + crop_size
                end_w = start_w + crop_size
                if is_label:
                    crop = img[start_h:end_h, start_w:end_w]
                else:
                    crop = img[:, start_h:end_h, start_w:end_w]
                if self.config.dataset.scale:
                    crop = self._rescale_to_power_of_two(crop, is_label)
                all_crops.append(crop)
        return all_crops




#NOTE: Given the from of loop these functions should be in any dataset module that you design (given that you keep it unchanged)
def train_set(config):
    return ExampleDataset(config)

def val_set(config):
    return ExampleDataset(config, part = 'val')

def test_set(config): 
    #Not properly implemented here or in above class
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



