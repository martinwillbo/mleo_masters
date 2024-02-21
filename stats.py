from torch.utils.data import DataLoader
import util
import torch
import numpy as np
from tqdm import tqdm

def stats(config):
    dataset_module = util.load_module(config.dataset.script_location)
    #NOTE: Just outlining the 'interface' of this way of structuring the experiment code
    train_set = dataset_module.train_set(config)
    train_loader = DataLoader(train_set, batch_size = config.batch_size, shuffle = False, num_workers = config.num_workers,
                            pin_memory = True)
    # Initialize variables for accumulating channel-wise sums
    #channel_sums = torch.zeros(10)
    #channel_squared_diff = torch.zeros(10)
    #num_batches = 0
    train_iter = iter(train_loader)
    num_img = 0
    senti_imgs = []
    for _, senti, _ in tqdm(train_iter):

        senti = senti.numpy()
        senti_batch = senti.reshape(-1, 10, 21, 21)    
        senti_imgs = np.concatenate((senti_batch, senti_imgs), axis = 0)

    mean = np.mean(senti_imgs, axis = (0,2,3))
    std = np.std(senti_imgs, axis = (0,2,3))
    print(mean)
    print(std)
    return mean, std