
import numpy as np
from torchvision.transforms import ColorJitter
import random
import torch

class BasicTransform():

    def __init__(self, config):
        self.config = config
        #all_means = np.load(config.norm_path)
        #self.layer_means = np.mean(all_means, axis=1)
        #self.layer_stds = np.std(all_means, axis = 1)
        if 'color-jitter' in self.config.transform.order:
            self.c_jitter = ColorJitter(
                brightness=config.transform.color_jitter_brightness,
                contrast=config.transform.color_jitter_contrast,
                saturation=config.transform.color_jitter_saturation,
                hue=(-config.transform.color_jitter_hue, config.transform.color_jitter_hue)
            )

    def apply(self, crop, gt_crop):
        #if self.config.save_first_batch:
            #org = crop.copy()
        for aug in self.config.transform.order:
            if aug == 'color-jitter':
                if random.random() < self.config.transform.p_color_jitter:
                    
                    print('this')
                    #TODO: Isn't this fucked up? Shouldnt we normalize after?
                    crop = self.c_jitter(torch.tensor(crop)).numpy()
            elif aug == 'up-down':
                if random.random() < self.config.transform.p_up_down_flip:
                    crop = np.flip(crop, axis=1)
                    gt_crop = np.flip(gt_crop, axis=0)
            elif aug == 'left-right':
                if random.random() < self.config.transform.p_left_right_flip:
                    crop = np.flip(crop, axis=2)
                    gt_crop = np.flip(gt_crop, axis=1)
            
        #NOTE: Negative stride is not supported by pytorch, so we need to copy the data.
        # I have no real idea what this means and why this fixes it, but if you stumble upon it this seems to fix it.
        crop = crop.copy()
        gt_crop = gt_crop.copy()

        #crop = np.transpose(crop, (1,2,0)).astype(float)
        #crop -= self.layer_means
        #crop /= self.layer_stds
        #crop = np.transpose(crop, (2,0,1))

        return crop, gt_crop
    
def get_transform(config):
    return BasicTransform(config)