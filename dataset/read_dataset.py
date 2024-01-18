from torch.utils.data import Dataset
from tqdm import tqdm
import tifffile
import numpy as np
import random
import torch
import os
import cv2
import math
class DatasetClass(Dataset):
    def __init__(self, config, part):
        self.config = config
        self.part = part
        self.X = []
        self.Y = []

        path_var = part
        if part == 'val':
            path_var = 'train'

        X_BASE_PATH = os.path.join(self.config.dataset.path, self.config.dataset.X_path + '_' + path_var)
        Y_BASE_PATH = os.path.join(self.config.dataset.path, self.config.dataset.Y_path + '_' + path_var)
        X_tif_paths = self._read_paths(X_BASE_PATH)
        Y_tif_paths = self._read_paths(Y_BASE_PATH)

        assert len(X_tif_paths) == len(Y_tif_paths)

        split_point = math.floor(len(X_tif_paths)*(1-self.config.dataset.val_set_size))

        if self.part == 'train':
            # no shuffle when splitting - change maybe
            X_tif_paths = X_tif_paths[0:split_point]
            Y_tif_paths = Y_tif_paths[0:split_point]
        if self.part == 'val':
            X_tif_paths = X_tif_paths[split_point:]
            Y_tif_paths = Y_tif_paths[split_point:]
        print('Constructing ' + self.part + ' set...')

        temp_X = self._read_data(X_tif_paths, is_label = False)
        temp_Y = self._read_data(Y_tif_paths, is_label = True)
        print(len(temp_X))
        print(len(temp_Y))
        self.X.extend(temp_X)
        self.Y.extend(temp_Y)
        print(min(item.min().item() for item in self.Y))
        print(max(item.max().item() for item in self.Y))
    
    def __getitem__(self, index):
        if self.config.dataset.using_priv:
            x = self.X[index]
        else:
            x = self.X[index][:3,:,:]
        y = self.Y[index]
        if self.part == 'val':
            return torch.tensor(x, dtype = torch.float), torch.tensor(y, dtype = torch.long)
        #if transforms, we need to add here
        return torch.tensor(x, dtype = torch.float), torch.tensor(y, dtype = torch.long)
    
    def __len__(self):
        assert len(self.X) == len(self.Y)
        return len(self.X)
    
    def _read_paths(self, BASE_PATH):
        tif_paths = []
        for root, dirs, files in os.walk(BASE_PATH):
            for file in files:
                if file.endswith(".tif"):
                    tif_paths.append(os.path.join(root, file))
        return tif_paths
    
    def _read_data(self, tif_paths, is_label):
        temp_data = []
        for i, path in tqdm(enumerate(tif_paths)):
            data = np.array(tifffile.imread(path))
            if is_label: #classes are 1 to 19, have to be 0 to 18
                data = data - 1 
            if not is_label:
                data = np.transpose(data, (2,0,1))
            if self.config.dataset.crop:
                data_list = self._crop(data, is_label)
                temp_data.extend(data_list)
            elif self.config.dataset.scale:
                data = self._rescale(data, is_label)
                temp_data.append(data)
            else:
                temp_data.append(data)
            if i == 200:
                return temp_data
            if i == 20 and self.part == 'val':
                return temp_data

        return temp_data

    def _rescale(self, data, is_label):
        if not is_label:
            data = np.transpose(data, (1,2,0))
        h, w = data.shape[0], data.shape[1]
        power_h = math.ceil(np.log2(h) / np.log2(2))
        power_w = math.ceil(np.log2(w) / np.log2(2))
        if 2**power_h != h or 2**power_w != w:
            if not is_label: #i.e. is normal image, guess it just depends on nbr channels
                data = cv2.resize(data, dsize=(2**power_h, 2**power_w), interpolation=cv2.INTER_CUBIC)
            else:
                data = cv2.resize(data, dsize=(2**power_h, 2**power_w), interpolation=cv2.INTER_NEAREST)
        if not is_label:
            data  = np.transpose(data, (2, 0, 1))
        return data
    def _crop(self, data, is_label):
        if is_label:
            h, w = data.shape[0], data.shape[1]
        else:
            h, w = data.shape[1], data.shape[2]
        crop_size = self.config.dataset.crop_size
        crop_step_size = self.config.dataset.crop_step_size
        all_crops = []
        for start_h in range(0,h - crop_size + 1, crop_step_size):
            for start_w in range(0,w - crop_size + 1, crop_step_size):
                end_h = start_h + crop_size
                end_w = start_w + crop_size
                if is_label:
                    crop = data[start_h:end_h, start_w:end_w]
                else:
                    crop = data[:, start_h:end_h, start_w:end_w]
                if self.config.dataset.scale:
                    crop = self._rescale(crop, is_label)
                all_crops.append(crop)
        return all_crops
    
#NOTE: Given the from of loop these functions should be in any dataset module that you design (given that you keep it unchanged)
def train_set(config):
    return DatasetClass(config, part = 'train')

def val_set(config):
    return DatasetClass(config, part = 'val')

def test_set(config):
    pass