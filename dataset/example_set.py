
#NOTE: This is just to illustrate how one could implement a dataset for EO. Tweak as needed.

from torch.utils.data import Dataset
from tqdm import tqdm
import tifffile
import numpy as np
import random
import torch
import os
import cv2
import math

#NOTE: Sometimes it's cleaner to just make two or three classes. For example ValExampleDataset
#NOTE: I would say it depends on how much they need to differ, if it's just a couple of lines if/else works fine.
class ExampleDataset(Dataset):

    def train_part_dummy(self, dataset_file_paths):
        #NOTE: Typically we want a fixed train/val split over experiments to keep things comparable, most datasets have this out-of-the-box.
        training_portion = None
        return training_portion

    def __init__(self, config, part = 'train'):
        self.config = config
        self.part = part
        self.transform = None

        #NOTE: This part nessecarily depends on your dataset structure. For this example I just assume all data points are just in one folder.
        #NOTE: I also assume we have the training portion of those points figured out through a dummy function.
        #NOTE: Further, this snippet also assumes we have enough RAM to read all of the files into memory, if this is not the case our
        # 'dataset' should typically consist of file paths, these we then index in the __getitem__ function and read into memory there.
        # This is waaaay slower and should typically only be done if we don't have RAM enough to read the data.
        # One can do a mix of both with some cleverness, threading, and custom tailoring of the Dataloader and Sampler modules but is overkill (also quite difficult) I think.
        dataset_file_paths = os.listdir(config.dataset.path)
        if self.part == 'train':
            point_paths = self.train_part_dummy(dataset_file_paths)
        else:
            pass

        all_means = np.load(config.norm_path)
        layer_means = np.mean(all_means, axis=1)
        self.layer_means = layer_means
        layer_stds = np.std(all_means, axis=1)
        self.layer_stds = layer_stds

        #NOTE: One can pre-allocate the needed memory using the np module (if not just storing file paths here), but it's really not nessecary, we only do this once.
        self.data = []
        self.labels = []
        print('Constructing '+self.part+' set...')
        for i, path in tqdm(enumerate(point_paths)):
            #NOTE: I assumed the files are .tiff, use whatever library that makes sense here, PIL, cv2 or others.
            #NOTE: Depending on the file etc sometimes the channels is first/last. You might need to use np.transpose at a couple of locations depending on what you do with the img.
            img = np.array(tifffile.imread(path))
            gt = np.array(tifffile.imread(path.replace('train', 'val')))
            if config.dataset.scale:
                #NOTE: For this to work (cv2.resize calls) channel needs to be the last dimension (H, W, C)
                h = img.shape[0]
                w = img.shape[1]
                power_h = math.ceil(np.log2(h) / np.log2(2))
                power_w = math.ceil(np.log2(w) / np.log2(2))
                if 2**power_h != h or 2**power_w != w:
                    img = cv2.resize(img, dsize=(2**power_h, 2**power_w), interpolation = cv2.INTER_CUBIC)
                    gt = cv2.resize(gt, dsize=(2**power_h, 2**power_w), interpolation = cv2.INTER_NEAREST)

            self.data.append(img)
            self.labels.append(gt)

    def set_transform(self, transform):
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        if self.part == 'val':
            #Typically we just convert the points and labels to tensors and return those
            pass

        #NOTE: Typically we have to balance batch_size/crop_size depending on VRAM limitations.
        #It could be better to do this in the __init__ function and add all possible crops of an image as a data point in self.data.
        #That way you would guarante that the model sees ALL data in every epoch. I think that would be prudent.
        start_h = random.randint(0, self.data[index].shape[1] - self.config.crop_size)
        end_h = start_h + self.config.crop_size
        start_w = random.randint(0, self.data[index].shape[2] - self.config.crop_size)
        end_w = start_w + self.config.crop_size
        crop = self.data[index][:, start_h:end_h, start_w:end_w]
        gt_crop = self.data_gt[index][start_h:end_h, start_w:end_w]

        #NOTE: I usually expect the transform to also normalize the data, but up to you.
        if self.tranform is not None:
            #NOTE: I typically like writing my own transformation instances like this, but you don't have to. Can for example use Compose from pytorch.
            #The downside is that you are stuck with already implemented functions and can't customize. But if it's not needed then using Compose is fine.
            crop, gt_crop = self.transform.apply(crop, gt_crop)
        else:
            #NOTE: These operations expect shape (H,W,C)
            crop = np.transpose(crop, (1,2,0)).astype(float)
            crop -= self.layer_means
            crop /= self.layer_stds
            #NOTE: Pytorch models typically expect shape (C, H, W)
            crop = np.transpose(crop, (2,0,1))
        
        return torch.tensor(crop, dtype = torch.float), torch.tensor(gt_crop, dtype = torch.long)


#NOTE: Given the from of loop these functions should be in any dataset module that you design (given that you keep it unchanged)
def train_set(config):
    return ExampleDataset(config)

def val_set(config):
    return ExampleDataset(config, part = 'val')

def test_set(config):
    pass