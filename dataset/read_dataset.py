from torch.utils.data import Dataset
from tqdm import tqdm
import tifffile
import datetime
import json
import numpy as np
import pandas as pd
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
        self.aerial_to_senti = {}

        #Read in desired transform
        transform_module = util.load_module(self.config.transform.script_location)
        self.transform = transform_module.get_transform(self.config)
        self.layer_means = np.array(self.config.dataset.mean)
        self.layer_stds = np.array(self.config.dataset.std)
        self.senti_layer_means = np.array(self.config.dataset.mean_senti)
        self.senti_layer_stds = np.array(self.config.dataset.std_senti)

        if not self.config.dataset.using_priv:
            self.layer_means = self.layer_means[0:3] #only bgr
            self.layer_stds = self.layer_stds[0:3]

        #only for trying with 4 channel
        if self.config.model.n_channels == 4:
            self.layer_means = self.layer_means[0:4] #only bgr  
            self.layer_stds = self.layer_stds[0:4]    
    
        if part == 'val' or part == 'train':       
            X_BASE_PATH = os.path.join(self.config.dataset.path, self.config.dataset.X_path + '_' + 'train')
            Y_BASE_PATH = os.path.join(self.config.dataset.path, self.config.dataset.Y_path + '_' + 'train')
            SENTI_BASE_PATH = os.path.join(self.config.dataset.path, self.config.dataset.senti_path + '_' + 'train')
        elif part == 'test':
            X_BASE_PATH = os.path.join(self.config.dataset.path, 'flair_2_aerial_test')
            Y_BASE_PATH = os.path.join(self.config.dataset.path, 'flair_2_labels_test')
            SENTI_BASE_PATH = os.path.join(self.config.dataset.path, 'flair_2_sen_test')

        if part == 'train' and config.dataset.dataset_size == 0.05:
            X_tif_paths = self._read_paths_from_file('../datasets/paths/X_paths_train_5.txt')
            Y_tif_paths = self._read_paths_from_file('../datasets/paths/Y_paths_train_5.txt')
            senti_data_paths = self._read_paths_from_file('../datasets/paths/senti_data_paths_train_5.txt')
            senti_mask_paths = self._read_paths_from_file('../datasets/paths/senti_mask_paths_train_5.txt')
            senti_dates_paths = self._read_paths_from_file('../datasets/paths/senti_dates_paths_train_5.txt')
        elif part == 'val' and config.dataset.dataset_size == 0.05:
            X_tif_paths = self._read_paths_from_file('../datasets/paths/X_paths_val_5.txt')
            Y_tif_paths = self._read_paths_from_file('../datasets/paths/Y_paths_val_5.txt')
            senti_data_paths = self._read_paths_from_file('../datasets/paths/senti_data_paths_val_5.txt')
            senti_mask_paths = self._read_paths_from_file('../datasets/paths/senti_mask_paths_val_5.txt')
            senti_dates_paths = self._read_paths_from_file('../datasets/paths/senti_dates_paths_val_5.txt')
        elif part == 'test' or config.dataset.dataset_size == 1.0:
            XX_tif_paths = self._read_paths(X_BASE_PATH, '.tif')
            Y_tif_paths = self._read_paths(Y_BASE_PATH, '.tif')
            senti_data_paths = self._read_paths(SENTI_BASE_PATH, "data.npy", X_BASE_PATH) # all aerial images within the same area have the same 
            senti_mask_paths = self._read_paths(SENTI_BASE_PATH, "masks.npy", X_BASE_PATH)# sentinel image so redundant to store one for each 
            senti_dates_paths = self._read_paths(SENTI_BASE_PATH, "products.txt", X_BASE_PATH) 

        aerial_to_senti_path = os.path.join(self.config.dataset.path, 'flair-2_centroids_sp_to_patch.json') # load the dictionary wwith mapping from sentinel to aerial patches
        with open(aerial_to_senti_path) as file:
            self.aerial_to_senti = json.load(file)

        combined = list(zip(X_tif_paths, Y_tif_paths, senti_data_paths, senti_mask_paths, senti_dates_paths))
        random.shuffle(combined)
        X_tif_paths, Y_tif_paths, senti_data_paths, senti_mask_paths, senti_dates_paths = zip(*combined)
        X_tif_paths, Y_tif_paths, senti_data_paths, senti_mask_paths, senti_dates_paths = list(X_tif_paths), list(Y_tif_paths), list(senti_data_paths), list(senti_mask_paths), list(senti_dates_paths)
        assert len(X_tif_paths) == len(Y_tif_paths) == len(senti_data_paths) == len(senti_mask_paths) == len(senti_dates_paths)

        if len(X_tif_paths) % config.batch_size == 1 and part == 'train':
            self.X_tif_paths = X_tif_paths[:-1]
            self.Y_tif_paths = Y_tif_paths[:-1]
            self.senti_data_paths = senti_data_paths[:-1]
            self.senti_mask_paths = senti_mask_paths[:-1]
            self.senti_dates_paths= senti_dates_paths[:-1]
        elif len(X_tif_paths) % config.val_batch_size == 1 and part == 'val':
            self.X_tif_paths = X_tif_paths[:-1]
            self.Y_tif_paths = Y_tif_paths[:-1]
            self.senti_data_paths = senti_data_paths[:-1]
            self.senti_mask_paths = senti_mask_paths[:-1]
            self.senti_dates_paths= senti_dates_paths[:-1]
        elif len(X_tif_paths) % config.test_batch_size == 1 and part == 'test':
            self.X_tif_paths = X_tif_paths[:-1]
            self.Y_tif_paths = Y_tif_paths[:-1]
            self.senti_data_paths = senti_data_paths[:-1]
            self.senti_mask_paths = senti_mask_paths[:-1]
            self.senti_dates_paths= senti_dates_paths[:-1]
        else:
            self.X_tif_paths = X_tif_paths
            self.Y_tif_paths = Y_tif_paths
            self.senti_data_paths = senti_data_paths
            self.senti_mask_paths = senti_mask_paths
            self.senti_dates_paths = senti_dates_paths
        
        # This is for determenistic cropping - curr. not used
            
        if config.dataset.det_crop and self.part == 'train': #THIS IS BROKEN WITH THE SHUFFLING
            self.crop_coordinates = self._get_crop_coordinates(self._read_data(X_tif_paths[0], is_label=False)) #use one img for cropping coords
            num_crops = len(self.crop_coordinates)
            self.X_tif_paths = [path for path in X_tif_paths for _ in range(num_crops)]
            self.Y_tif_paths = [path for path in Y_tif_paths for _ in range(num_crops)]

        

    def __getitem__(self, index):
       
        x = self._read_data(self.X_tif_paths[index], is_label = False)
        y = self._read_data(self.Y_tif_paths[index], is_label = True)
        senti = self._read_senti_patch(self.senti_data_paths[index], self.senti_mask_paths[index], self.X_tif_paths[index]) # this takes the data and masks and concatinates along dim=1
        dates = self._read_dates(self.senti_dates_paths[index])           

        if self.part == 'val' or self.part == 'test':
            x = self._normalize(x)
            senti = self._normalize_senti(senti)
            monthly_senti = self._monthly_image(senti, dates)

            return torch.tensor(x, dtype = torch.float), torch.tensor(y, dtype = torch.long), torch.tensor(monthly_senti, dtype= torch.float)
        
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
            x, y, senti = self.transform.apply(x,y, senti)
           
        #NOTE: These operations expect shape (H,W,C)
            
        #Normalize images, after transform
        x = self._normalize(x)    
        senti = self._normalize_senti(senti)
        monthly_senti = self._monthly_image(senti, dates)
        
        return torch.tensor(x, dtype = torch.float), torch.tensor(y, dtype = torch.long), torch.tensor(monthly_senti, dtype = torch.float)
    
    def __len__(self):
        assert len(self.X_tif_paths) == len(self.Y_tif_paths)
        return len(self.X_tif_paths)
    
    
    def _read_paths(self, BASE_PATH, ending, X_path = None):
        
        paths = []
        if ending == '.tif':
            for root, dirs, files in os.walk(BASE_PATH):
                for file in sorted(files):
                    if file.endswith(ending): # should not be necessary
                        paths.append(os.path.join(root, file))

        elif ending == 'data.npy' or ending == 'masks.npy' or ending == 'products.txt':
    
            #Stores the number of times a sentinel image/mask/time series info should be added to path list
            area_counts = self._count_files(X_path)

            for root, dirs, files in os.walk(BASE_PATH):
                for file in sorted(files):
                        if file.endswith(ending):

                            path_split = root.split(os.sep)
                            area_name = os.sep.join(path_split[-3:-1]) #domain + area name - should be unique
                            count = area_counts[area_name]
                            
                            for i in range(count):
                                paths.append(os.path.join(root, file))         
                      
        return paths
    
    def _read_paths_from_file(self, file_path):
        with open(file_path, 'r') as file:
            paths = [line.strip() for line in file.readlines()]
        return paths
    
    def _count_files(self, base_path):

        #This function counts files in subdirectories
        aerial_counts = {}
        
        for root, dirs, files in os.walk(base_path):
            if len(files) > 0:
                path_sections = root.split(os.sep)  # Split the path using the separator
                area_name = os.sep.join(path_sections[-3:-1])  # Join 
        
                aerial_counts[area_name] = len(files)

        return aerial_counts

    
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
            if self.config.model.n_channels == 4:
                    data = data[:4,:,:] 
        return data
    
    def _read_senti_patch(self, data_path, mask_path, X_path): 

        data = np.load(data_path) #T x C x H x W
        mask = np.load(mask_path) #T x 2 x H x W
        data = data.astype(np.uint8)
        mask = mask.astype(np.uint8)

        #Extract image index
        filename = os.path.basename(X_path)
        image_index = filename.split('/')[-1]

        #Get centroid
        x_cent, y_cent = self.aerial_to_senti[image_index]
       
        #Extract patch
        side = self.config.dataset.senti_size
        data = data[:,:, x_cent-side:x_cent+side+1, y_cent-side:y_cent+side+1]
        mask = mask[:,:, x_cent-side:x_cent+side+1, y_cent-side:y_cent+side+1]

        data = np.concatenate((data, mask), axis=1)
        
        return data  
    
    def _read_dates(self, txt_file):

        #Open text file with date info
        with open(txt_file, 'r') as f:
            products= f.read().splitlines()
        
        dates = []

        #save the data for each senti patch
        for file in products:
            dates.append(datetime.datetime(int(file[11:15]), int(file[15:19][:2]), int(file[15:19][2:]))) #year not relevant but expected

        return np.array(dates)

    
    def _monthly_image(self, senti_patch, senti_raw_dates):

        #masks last 2 channels
        mask = senti_patch[:,-2:,:,:]
        patches = senti_patch[:,:-2,:,:]

        #filter out dates
        dates_to_keep = self._filter_dates(mask, area_threshold=0.5, proba_threshold=60)
        dates = senti_raw_dates[dates_to_keep]

        #filter masks and patches to use
        mask = mask[dates_to_keep,:,:,:]
        patches = patches[dates_to_keep,:,:,:]

        #Initialize
        mean_patches = []
        prev_mean = None

        # calc mean for each month
        for m in range(1,13):

            month_dates = list(filter(lambda i: (dates[i].month == m), range(len(dates))))

            if len(month_dates)!=0:
                prev_mean = np.mean(patches[month_dates,:,:,:], axis=0)
                mean_patches.append(prev_mean)
                
            else: 
                if prev_mean is not None:
                    mean_patches.append(prev_mean)
                else:
                    #print('No previous data, zero_padding instead')
                    mean_patches.append(np.zeros((10, 2*self.config.dataset.senti_size + 1, 2*self.config.dataset.senti_size + 1)))               
                      

        return np.array(mean_patches)
        
        
    
    def _filter_dates(self, mask, area_threshold, proba_threshold):
        
        dates_to_keep = []
        
        for t in range(mask.shape[0]):
            cover = np.count_nonzero((mask[t, 0, :,:]>=proba_threshold)) + np.count_nonzero((mask[t, 1, :,:]>=proba_threshold))
            cover /= mask.shape[2]*mask.shape[3]
            if cover < area_threshold:
                dates_to_keep.append(t)

        return dates_to_keep
        
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
    
    def _normalize(self, data):
        data = np.transpose(data, (1,2,0)).astype(float)
        data -= self.layer_means
        data /= self.layer_stds
        #NOTE: Pytorch models typically expect shape (C, H, W)
        data = np.transpose(data, (2,0,1))
        return data
    
    def _normalize_senti(self, senti):
        
        #separate data and mask
        mask = senti[:,-2:,:,:]
        data = senti[:,:-2,:,:]

        # Transpose data 
        data = np.transpose(data, (0, 2, 3, 1)).astype(float)
        
        # Subtract mean values
        data -= self.senti_layer_means
        
        # Divide by standard deviation values
        data /= self.senti_layer_stds
        
        # Transpose back 
        data = np.transpose(data, (0, 3, 1, 2))
        senti = np.concatenate((data, mask), axis=1)

        return senti
    
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
    return DatasetClass(config, part = 'test')