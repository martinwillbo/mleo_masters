from torch.utils.data import DataLoader
import util
def stats(config):
    dataset_module = util.load_module(config.dataset.script_location)
    #NOTE: Just outlining the 'interface' of this way of structuring the experiment code
    train_set = dataset_module.train_set(config)
    train_loader = DataLoader(train_set, batch_size = config.train_batch_size, shuffle = False, num_workers = config.num_workers,
                            pin_memory = True)
    # Initialize variables for accumulating channel-wise sums
    channel_sums = torch.zeros(3)
    channel_squared_diff = torch.zeros(3)
    for X, _ in train_loader:
        batch_size = X.size(0)
        num_batches += 1
        X = X[:, :3, :, :]
        # Calculate channel-wise sums and squared sums
        channel_sums += X.sum(dim=(0, 2, 3))
    # Calculate mean and standard deviation across all batches
    mean = channel_sums / (num_batches * batch_size * 512 * 512)
    for X, _ in train_loader:
        X = X[:, :3, :, :]
        X_diff = X - channel_sums
        channel_squared_diff += (X_diff**2).sum(dim=(0, 2, 3))
    variance = (channel_squared_diff / (batch_size * num_batches * 512 * 512))
    std = torch.sqrt(variance)
    print(f"Mean across channels: {mean}")
    print(f"Standard deviation across channels: {std}")
    return mean, std