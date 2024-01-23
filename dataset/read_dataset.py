from torch.utils.data import Dataset
from tqdm import tqdm
import tifffile
import numpy as np
import random
import torch
import os
import cv2
import math
import sys
import util
import random
class DatasetClass(Dataset):
    def __init__(self, config, part):
        self.config = config
        self.part = part
        self.X = []
        self.Y = []
        #Read in desiread transform
        transform_module = util.load_module(self.config.transform.script_location)
        self.transform = transform_module.get_transform(self.config)
        self.layer_means = np.array(self.config.dataset.mean)
        self.layer_stds = np.array(self.config.dataset.std)
        if not self.config.dataset.using_priv:
            self.layer_means = self.layer_means[0:3] #only bgr
            self.layer_stds = self.layer_stds[0:3]
        path_var = part
        if part == 'val':
            path_var = 'train'
        X_BASE_PATH = os.path.join(self.config.dataset.path, self.config.dataset.X_path + '_' + path_var)
        Y_BASE_PATH = os.path.join(self.config.dataset.path, self.config.dataset.Y_path + '_' + path_var)
        print(X_BASE_PATH)
        
        X_tif_paths = self._read_paths(X_BASE_PATH)
        Y_tif_paths = self._read_paths(Y_BASE_PATH)
        combined = list(zip(X_tif_paths, Y_tif_paths))
        random.shuffle(combined)
        X_tif_paths, Y_tif_paths = zip(*combined)
        X_tif_paths, Y_tif_paths = list(X_tif_paths), list(Y_tif_paths)
        assert len(X_tif_paths) == len(Y_tif_paths)
        split_point = math.floor(len(X_tif_paths)*(1-self.config.dataset.val_set_size))
        if self.part == 'train':
            # no shuffle when splitting - change maybe
            X_tif_paths = X_tif_paths[0:split_point]
            Y_tif_paths = Y_tif_paths[0:split_point]
        if self.part == 'val':
            X_tif_paths = X_tif_paths[split_point:]
            Y_tif_paths = Y_tif_paths[split_point:]
        #print('Constructing ' + self.part + ' set...')
        if config.dataset.det_crop: #THIS IS BROKEN WITH THE SHUFFLING
            self.crop_coordinates = self._get_crop_coordinates(self._read_data(X_tif_paths[0], is_label=False)) #use one img for cropping coords
            num_crops = len(self.crop_coordinates)
            self.X_tif_paths = [path for path in X_tif_paths for _ in range(num_crops)]
            self.Y_tif_paths = [path for path in Y_tif_paths for _ in range(num_crops)]
        else:
            self.X_tif_paths = X_tif_paths
            self.Y_tif_paths = Y_tif_paths
        print('Tif size: ' + str(sys.getsizeof(self.X_tif_paths)*8)) #takes like 3MB
        #temp_X = self._read_data_old(X_tif_paths, is_label = False)
        #temp_Y = self._read_data_old(Y_tif_paths, is_label = True)
        #print(len(temp_X))
        #print(len(temp_Y))
        #self.X.extend(temp_X)
        #self.Y.extend(temp_Y)
        #print(min(item.min().item() for item in self.Y))
        #print(max(item.max().item() for item in self.Y))
    def __getitem__(self, index):
        #print(index)
        x = self._read_data(self.X_tif_paths[index], is_label = False)
        #x = self.X[index]
        y = self._read_data(self.Y_tif_paths[index], is_label = True)
        #y = self.Y[index]
        if self.part == 'val':
            return torch.tensor(x, dtype = torch.float), torch.tensor(y, dtype = torch.long)
        if self.config.dataset.det_crop:
            #get exactly one crop
            crop_coords = self.crop_coordinates[index % len(self.crop_coordinates)]
            x = x[:, crop_coords[0]:crop_coords[2], crop_coords[1]:crop_coords[3]]
            y = y[crop_coords[0]:crop_coords[2], crop_coords[1]:crop_coords[3]]
        elif self.config.dataset.random_crop:
            x,y = self._random_crop(x,y)
        if self.config.dataset.scale:
            x = self._rescale(x, is_label = False)
            y = self._rescale(y, is_label = True)
        if self.config.use_transform:            
            x, y = self.transform.apply(x,y)
           
        #NOTE: These operations expect shape (H,W,C)
        #Normalize images, after transform
        x = np.transpose(x, (1,2,0)).astype(float)
        x -= self.layer_means
        x /= self.layer_stds
        #NOTE: Pytorch models typically expect shape (C, H, W)
        x = np.transpose(x, (2,0,1))
        
        return torch.tensor(x, dtype = torch.float), torch.tensor(y, dtype = torch.long)
    def __len__(self):
        assert len(self.X_tif_paths) == len(self.Y_tif_paths)
        return len(self.X_tif_paths)
    def _read_paths(self, BASE_PATH):
        tif_paths = []
        for root, dirs, files in os.walk(BASE_PATH):
            for file in files:
                if file.endswith(".tif"):
                    tif_paths.append(os.path.join(root, file))
        return tif_paths
    def _read_data(self, tif_path, is_label):
        data = np.array(tifffile.imread(tif_path))
        data = data.astype(np.uint8) #all data is uint8
        if is_label: #classes are 1 to 19, have to be 0 to 18
            if self.config.model.n_class < 19: #group last classes as in challenge
                data[data > 12] = 13
            data = data - 1
        if not is_label:
            data = np.transpose(data, (2,0,1))
            if not self.config.dataset.using_priv:
                data = data[:3,:,:]
        return data
    def _read_data_old(self, tif_paths, is_label):
        temp_data = []
        for i, path in tqdm(enumerate(tif_paths)):
            data = np.array(tifffile.imread(path))
            data = data.astype(np.uint8) #all data is uint8
            if is_label: #classes are 1 to 19, have to be 0 to 18
                if self.config.model.n_class < 19: #group last classes as in challenge
                    data[data > 12] = 13
                data = data - 1
            if not is_label:
                data = np.transpose(data, (2,0,1))
                if not self.config.dataset.using_priv:
                    data = data[:3,:,:]
            if self.config.dataset.det_crop:
                data_list = self.det_crop_old(data, is_label)
                temp_data.extend(data_list)
            elif self.config.dataset.scale:
                data = self._rescale(data, is_label)
                temp_data.append(data)
            else:
                temp_data.append(data)
            #if i == 20:
            #    return temp_data
            #if i == 2 and self.part == 'val':
            #    return temp_data
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
    def _get_crop_coordinates(self, data):
        h, w = data.shape[1], data.shape[2] #x,y has the same shape
        crop_size = self.config.dataset.crop_size
        crop_step_size = self.config.dataset.crop_step_size
        crop_coordinates = []
        for start_h in range(0, h - crop_size + 1, crop_step_size):
            for start_w in range(0, w - crop_size + 1, crop_step_size):
                end_h = start_h + crop_size
                end_w = start_w + crop_size
                # Store the crop coordinates as a tuple
                coordinates = (start_h, start_w, end_h, end_w)
                crop_coordinates.append(coordinates)
        return crop_coordinates
    def _random_crop(self, x, y):
        start_h = random.randint(0, x.shape[1] - self.config.dataset.crop_size)
        end_h = start_h + self.config.dataset.crop_size
        start_w = random.randint(0, x.shape[2] - self.config.dataset.crop_size)
        end_w = start_w + self.config.dataset.crop_size
        x = x[:, start_h:end_h, start_w:end_w]
        y = y[start_h:end_h, start_w:end_w]
        return x,y
    def det_crop_old(self, data, is_label):
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