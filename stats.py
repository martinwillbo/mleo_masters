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
    channel_sums = torch.zeros(10)
    channel_squared_diff = torch.zeros(10)
    #num_batches = 0
    train_iter = iter(train_loader)
    num_img = 0

    for _, senti, _ in tqdm(train_iter):
        #dim will be batch x month x channel x height x
        num_img += len(senti)
        senti = torch.mean(senti, dim=1)
        #print(senti.mean(dim=(0,2,3)))
        # Calculate channel-wise sums and squared sums
        channel_sums += senti.sum(dim=(0, 2, 3))
    # Calculate mean and standard deviation across all batches
    mean = channel_sums / (num_img * 21 * 21)
    print(mean)

    train_iter = iter(train_loader)
    for _, senti, _ in tqdm(train_iter):

        # Assuming 'batch' is your batch tensor with dimensions (batch_size, 12, 21, 21)

        # Sum pixel values along the width and height dimensions (2, 3) for each image
        pixel_sum_per_image = torch.sum(senti, dim=(2, 3))

        # Check if the pixel sum for each image within each item is zero
        is_zero_sum_image = torch.sum(pixel_sum_per_image, dim=1) == 0

        # Find indices where the pixel sum for any image within each item is not zero
        non_zero_indices_per_item = torch.nonzero(is_zero_sum_image == False)

        # Remove images with zero pixel sum within each item
        senti = torch.index_select(senti, 0, non_zero_indices_per_item.squeeze(1))
        
        senti = torch.mean(senti, dim=1)
        #print(X.mean(dim=(0,2,3)))
        channel_reshaped = mean.view(1, 5, 1, 1)
        _diff = senti - channel_reshaped
        print(torch.mean(_diff, dim=(0,2,3)))
        channel_squared_diff += (_diff**2).sum(dim=(0, 2, 3))
    
    variance = (channel_squared_diff / (num_img * 21 * 21))
    std = torch.sqrt(variance)
    print(f"Mean across channels: {mean}")
    print(f"Standard deviation across channels: {std}")
    return mean, std