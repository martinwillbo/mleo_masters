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
    
    batch_size = 8
    num_months = 12
    num_channels = 10
    image_height = 21
    image_width = 21
    count = 0
    channel_lists = [[] for _ in range(num_channels)]  # Create empty lists for each channel
    for _, senti, _ in tqdm(train_iter):        
            # Iterate through months
            senti = np.array(senti)
            for month in senti:
                # Iterate through images in the month
                for image in month:
                        # Iterate through channels
                        for channel_index in range(num_channels):
                            # Extract pixels for the current channel
                            pixels = image[channel_index].flatten()
                            # Append pixels to the channel list
                            print(pixels.dtype)
                            if(np.sum(pixels) > 0):
                                  channel_lists[channel_index].extend(pixels)
                            else:
                                  count += 1

                    

    mean = np.mean(channel_lists)
    std = np.std(channel_lists)

    print(mean)
    print(std)
    print('count: ' + str(count))

    return mean, std