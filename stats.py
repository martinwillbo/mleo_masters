from torch.utils.data import DataLoader
import util
import torch
from tqdm import tqdm

def stats(config):
    dataset_module = util.load_module(config.dataset.script_location)
    #NOTE: Just outlining the 'interface' of this way of structuring the experiment code
    train_set = dataset_module.train_set(config)
    train_loader = DataLoader(train_set, batch_size = config.batch_size, shuffle = False, num_workers = config.num_workers,
                            pin_memory = True)
    # Initialize variables for accumulating channel-wise sums
    channel_sums = torch.zeros(5)
    channel_squared_diff = torch.zeros(5)
    #num_batches = 0
    train_iter = iter(train_loader)
    num_img = 0

    for X, _ in tqdm(train_iter):
        num_img += len(X)
        print(X.mean(dim=(0,2,3)))
        # Calculate channel-wise sums and squared sums
        channel_sums += X.sum(dim=(0, 2, 3))
    # Calculate mean and standard deviation across all batches
    mean = channel_sums / (num_img * 512 * 512)
    print(mean)

    train_iter = iter(train_loader)
    for X, _ in tqdm(train_iter):
        #DOES NOT WORK
        X = X[:, :, :, :]
        print(X.mean(dim=(0,2,3)))
        channel_reshaped = mean.view(1, 5, 1, 1)
        X_diff = X - channel_reshaped
        print(X_diff.mean(dim=(0,2,3)))
        channel_squared_diff += (X_diff**2).sum(dim=(0, 2, 3))
    
    variance = (channel_squared_diff / (num_img * 512 * 512))
    std = torch.sqrt(variance)
    print(f"Mean across channels: {mean}")
    print(f"Standard deviation across channels: {std}")
    return mean, std

import json
import statistics

def metadata_stats():
    with open('/raid/aleksispi/master-theses/agnes-malte-spring2024/flair_aerial_metadata.json', 'r') as f: 
        metadata_dict = json.load(f)

    val_set_doms = ["D004", "D014", "D029", "D031", "D058", "D066", "D067", "D077"]
    test_set_doms = ["D015", "D026", "D061", "D068", "D071", "D022", "D036", "D064", "D069", "D084"]
    x_coord_list = []
    y_coord_list = []
    alt_list = []
    date_list = []
    time_list = []
    for img in metadata_dict:
        dom = metadata_dict[img]["domain"][0:4]
        if dom not in val_set_doms and dom not in test_set_doms: 
            x_coord_list.append(metadata_dict[img]["patch_centroid_x"])
            y_coord_list.append(metadata_dict[img]["patch_centroid_y"])
            alt_list.append(metadata_dict[img]["patch_centroid_z"])
            
            date = metadata_dict[img]["date"]
            _, mm, dd = date.split('-')
            date_list.append( int(mm)*30 + int(dd))

            time = metadata_dict[img]["time"]
            hh = int(time.split('h')[0]) 
            mm = int(time.split('h')[1]) 
            time_list.append(hh*60 + mm)

    print(statistics.mean(x_coord_list), statistics.mean(y_coord_list), statistics.mean(alt_list), statistics.mean(date_list), statistics.mean(time_list))
    print(statistics.stdev(x_coord_list), statistics.stdev(y_coord_list), statistics.stdev(alt_list), statistics.stdev(date_list), statistics.stdev(time_list))

    

metadata_stats()

      
