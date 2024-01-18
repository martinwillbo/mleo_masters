import numpy as np
import torch
import random
import loop
import yaml
from dataset.read_set import train_set, val_set, ExampleDataset
import torch
import torch.nn as nn
import torch.nn.functional as F



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
    def __getattr__(self, key):
        if key in self:
            item = self[key]
            if isinstance(item, dict):
                return DotDict(item)
            elif isinstance(item, list):
                return [DotDict(i) if isinstance(i, dict) else i for i in item]
            return item
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
             
config_path_dataset = 'config/dataset/read_set.yaml'
config_data_dataset = load_config(config_path_dataset)

config_path ='config/config.yaml'
config_data = load_config(config_path)

config_path_transform = 'config/transform/basic.yaml'
config_data_transform = load_config(config_path_transform)

config_data['dataset'] = config_data_dataset
config_data['transform'] = config_data_transform

config = DotDict(config_data)
print(config.dataset)

#Use dataloader, which has builtin features for not loading all the data to the heap simultaniously, but only one batch at the time
#train_set = train_set(config_dataset)
#val_set = val_set(config_dataset)

#print(train_set[0][0].shape) #first index is image, second is input v.s. label, then [512, 512, 5]


random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed(config.seed)

class TinyCNN(nn.Module):
    def __init__(self, num_classes):
        super(TinyCNN, self).__init__()
        # Define the layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, num_classes, kernel_size=1)  # 1x1 convolution to get class scores

    def forward(self, x):
        # Forward pass
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)  # Output is [N, C, H, W]
        return {'out': x}

# Create the CNN model
model = TinyCNN(config.model.n_class)

#NOTE: Don't have to use tensorboard to log experiments, but should implement something else if so (Aleksis have code for this).
#loop.loop(config, model)