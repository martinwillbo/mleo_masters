import torch
import numpy as np
import torch.nn as nn


def salt_n_pepper(x_priv, noise_level):
    B, C, H, W = x_priv.shape
    num_pixels_to_modify = int(noise_level * B * C * H * W)

    # Create a mask for pixels to modify
    indices = torch.randperm(B * C * H * W)[:num_pixels_to_modify]  # Get random indices
    mask = torch.zeros(B * C * H * W, dtype=torch.bool)
    mask[indices] = True  # Set selected indices to True
    mask = mask.view(B, C, H, W)  # Reshape mask to match image dimensions

    #set "black" and "white" values; mean 0, std 1
    black_value = 5 #np.array(config.dataset.mean)[channel_num] + 3*np.array(config.dataset.std)[channel_num]
    white_value = -5 #np.array(config.dataset.mean)[channel_num] - 3*np.array(config.dataset.std)[channel_num]# Randomly choose between black_value and white_value for the selected pixels
    
    #Randomly choose between black_value and white_value for the selected pixels
    black_or_white = torch.rand(H, W) < 0.5
    black_or_white = black_or_white * (white_value - black_value) + black_value

    # Apply the noise to the image
    x_priv_noise = torch.where(mask, black_or_white, x_priv)
    return x_priv_noise

def pixel_wise_fade(x_priv, noise_level):
    B, C, H, W = x_priv.shape
    num_pixels_to_modify = int(noise_level * B * C * H * W)

    # Create a mask for pixels to modify
    indices = torch.randperm(B * C * H * W)[:num_pixels_to_modify]  # Get random indices
    mask = torch.zeros(B * C * H * W, dtype=torch.bool)
    mask[indices] = True  # Set selected indices to True
    mask = mask.view(B, C, H, W)  # Reshape mask to match image dimensions

    # Generate random values for the selected pixels
    #mean = np.array(config.dataset.mean)[channel_num]
    #std = np.array(config.dataset.std)[channel_num]
    #data is normalized
    random_values = torch.normal(0, 1, size=(num_pixels_to_modify,))

    random_values_image = torch.zeros_like(x_priv)
    random_values_image[mask] = random_values

    # Apply the random values to the image
    x_priv_noise = torch.where(mask, random_values_image, x_priv)

    return x_priv_noise

def image_wise_fade(x_priv, noise_level):
    B, C, H, W = x_priv.shape
    #mean = np.array(config.dataset.mean)[channel_num]
    #std = np.array(config.dataset.std)[channel_num]
    random_image = torch.normal(0, 1, size=(B,C,H,W))

    x_priv_noise = (1-noise_level)*x_priv + noise_level*random_image

    return x_priv_noise

def zero_out(noise_level, model, three_five=False):
    with torch.no_grad():
        stages = model.encoder.get_stages()
        # The first convolutional layer is part of the second stage (index 1) in the list
        # which is an nn.Sequential containing _conv_stem, _bn0, _swish
        first_conv_layer = stages[1][0].weight.data  # Accessing _conv_stem directly within nn.Sequential
        mask_shape = first_conv_layer[:, 3:5, :, :].shape
        # Create a mask with noise_level amount of zeros
        mask = torch.bernoulli(torch.full(mask_shape, 1 - noise_level)).to(first_conv_layer.device)
        # Apply the mask
        first_conv_layer[:, 3:5, :, :] *= mask
        if three_five:
            first_conv_layer[:, 0:3, :, :] *= 5/3 #size up weights
    return model

def stepwise_linear_function_1(x, max_epochs):
    print("ABORT")
    if x/max_epochs < 100:
        return 0.3 * x / 100
    elif 100 <= x < 900:
        return 0.3 + (0.7 / 800) * (x - 100)
    else:
        return 1.0
    

def stepwise_linear_function_2(epoch, max_epochs):
    if epoch <= 0.1 * max_epochs:
        # Linear interpolation from 0 to 0.1 over 0 to 0.1*max_epochs
        return 0.1 / (0.1 * max_epochs) * epoch
    elif epoch <= 0.2 * max_epochs:
        # Linear interpolation from 0.1 to 0.5 over 0.1*max_epochs to 0.2*max_epochs
        return 0.1 + (0.4 / (0.1 * max_epochs)) * (epoch - 0.1 * max_epochs)
    elif epoch <= 0.9 * max_epochs:
        # Linear interpolation from 0.5 to 1.0 over 0.2*max_epochs to 0.9*max_epochs
        return 0.5 + (0.5 / (0.7 * max_epochs)) * (epoch - 0.2 * max_epochs)
    else:
        # Constant value of 1.0 beyond 0.9*max_epochs
        return 1.0
    
def custom_sine(x):
    amplitude = 0.5
    period = 100
    phase_shift = np.pi / 2  # Shifts the start to 0.5 and makes it go upwards
    vertical_shift = 0.5
    
    # Sine function formula adjusted for specified conditions
    return amplitude * np.sin(2 * np.pi * (x/period) + phase_shift) + vertical_shift


def set_noise(config, x, noise_level, noise_type):
    if noise_type == "salt_n_pepper":
        x[:, 3:5, :, :] = salt_n_pepper(x[:, 3:5, :, :], noise_level)

    elif noise_type == "pixel_wise_fade":
        x[:, 3:5, :, :] = pixel_wise_fade(x[:, 3:5, :, :], noise_level)
    
    elif noise_type == "image_wise_fade":
        x[:, 3:5, :, :] = image_wise_fade(x[:, 3:5, :, :], noise_level)
    
    else:
        print("Invalid noise_type")

    return x