import torch
import numpy as np


def salt_n_pepper(config, x_priv_layer, noise_level, channel_num):
    H, W = x_priv_layer.shape
    num_pixels_to_modify = int(noise_level * H * W)

    # Create a mask for pixels to modify
    indices = torch.randperm(H * W)[:num_pixels_to_modify]  # Get random indices
    mask = torch.zeros(H * W, dtype=torch.bool)
    mask[indices] = True  # Set selected indices to True
    mask = mask.view(H, W)  # Reshape mask to match image dimensions

    #set "black" and "white" values
    black_value = np.array(config.dataset.mean)[channel_num] + 3*np.array(config.dataset.std)[channel_num]
    white_value = np.array(config.dataset.mean)[channel_num] - 3*np.array(config.dataset.std)[channel_num]# Randomly choose between black_value and white_value for the selected pixels
    
    #Randomly choose between black_value and white_value for the selected pixels
    black_or_white = torch.rand(H, W) < 0.5
    black_or_white = black_or_white * (white_value - black_value) + black_value

    # Apply the noise to the image
    x_priv_noise = torch.where(mask, black_or_white, x_priv_layer)
    return x_priv_noise

def pixel_wise_fade(config, x_priv_layer, noise_level, channel_num):
    H,W = x_priv_layer.shape
    num_pixels_to_modify = int(noise_level * H * W)

    # Create a mask for pixels to modify
    indices = torch.randperm(H * W)[:num_pixels_to_modify]  # Get random indices
    mask = torch.zeros(H * W, dtype=torch.bool)
    mask[indices] = True  # Set selected indices to True
    mask = mask.view(H, W)  # Reshape mask to match image dimensions

    # Generate random values for the selected pixels
    mean = np.array(config.dataset.mean)[channel_num]
    std = np.array(config.dataset.std)[channel_num]
    random_values = torch.normal(mean, std, size=(num_pixels_to_modify,))

    random_values_image = torch.zeros_like(x_priv_layer)
    random_values_image[mask] = random_values

    # Apply the random values to the image
    x_priv_noise = torch.where(mask, random_values_image, x_priv_layer)

    return x_priv_noise

def image_wise_fade(config, x_priv_layer, noise_level, channel_num):
    H, W = x_priv_layer.shape
    mean = np.array(config.dataset.mean)[channel_num]
    std = np.array(config.dataset.std)[channel_num]
    random_image = torch.normal(mean, std, size=(H,W))

    x_priv_noise = (1-noise_level)*x_priv_layer + noise_level*random_image

    return x_priv_noise

def batch_chaos(config, x_priv, noise_level):
    #Use the method that seems to work the best, and do noise based on batch, not relevant here
    return x_priv

def set_noise(config, x, noise_level, noise_type):
    if noise_type == "salt_n_pepper":
            for i in range(len(x)):
                x[i,3,:,:] = salt_n_pepper(config, x[i,3, :, :], noise_level, 3)
                x[i,4,:,:] = salt_n_pepper(config, x[i,4, :, :], noise_level, 4)

    if noise_type == "pixel_wise_fade":
        for i in range(len(x)):
            x[i,3,:,:] = pixel_wise_fade(config, x[i,3, :, :], noise_level, 3)
            x[i,4,:,:] = pixel_wise_fade(config, x[i,4, :, :], noise_level, 4)

    if noise_type == "image_wise_fade":
        for i in range(len(x)):
            x[i,3,:,:] = image_wise_fade(config, x[i,3, :, :], noise_level, 3)
            x[i,4,:,:] = image_wise_fade(config, x[i,4, :, :], noise_level, 4)

    return x
