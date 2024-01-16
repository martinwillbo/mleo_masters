import numpy as np
import torch
import random
import loop
import yaml
from dataset.read_set import train_set, val_set, ExampleDataset



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

config_dataset = DotDict(config_data_dataset)
config = DotDict(config_data)

train_set = train_set(config_dataset)
val_set = val_set(config_dataset)

#print(train_set[0][0].shape) #first index is image, second is input v.s. label, then [512, 512, 5]


random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed(config.seed)

#NOTE: Don't have to use tensorboard to log experiments, but should implement something else if so (Aleksis have code for this).
loop.loop(config, train_set, val_set)