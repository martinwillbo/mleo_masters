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


