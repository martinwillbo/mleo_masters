import torch
import numpy as np


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

def zero_out(noise_level, model):
    with torch.no_grad():
        conv1_weights = model.backbone.conv1.weight
        mask_shape = conv1_weights[:, 3:5, :, :].shape

        # Create a mask with noise_level amount of zeros
        mask = torch.bernoulli(torch.full(mask_shape, 1 - noise_level)).to(conv1_weights.device)

        # Apply the mask
        conv1_weights[:, 3:5, :, :] *= mask

    return model

def stepwise_linear_function(x, max_epochs):
    if x < 100:
        return 0.3 * x / 100
    elif 100 <= x < max_epochs - 100:
        return 0.3 + (0.7 / 450) * (x - 100)
    else:
        return 1.0


def set_noise(config, x, noise_level, noise_type):
    if noise_type == "salt_n_pepper":
        x[:, 3:5, :, :] = salt_n_pepper(x[:, 3:5, :, :], noise_level)

    elif noise_type == "pixel_wise_fade":
        x[:, 3:5, :, :] = pixel_wise_fade(x[:, 3:5, :, :], noise_level)
    
    elif noise_type == "image_wise_fade":
        x[:, 3:5, :, :] = image_wise_fade(x[:, 3:5, :, :], noise_type)

    return x
