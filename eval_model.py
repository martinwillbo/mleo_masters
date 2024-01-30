import torch
import torch.nn as nn
from loop2 import miou_prec_rec_writing, miou_prec_rec_writing_13, conf_matrix
from torch.utils.data import DataLoader
import util
from torchvision.models.segmentation.deeplabv3 import deeplabv3_resnet50
import os
import tqdm
import numpy as np


def eval_model(config, writer, training_path, eval_type):
    dataset_module = util.load_module(config.dataset.script_location)

    val_set = dataset_module.val_set(config)
    val_loader = DataLoader(val_set, batch_size = config.val_batch_size, shuffle = False, num_workers = config.num_workers,
                            pin_memory = True)


    #Initialize model
    model = deeplabv3_resnet50(weights = config.model.pretrained, progress = True, #num_classes = config.model.n_class,
                                dim_input = config.model.n_channels, aux_loss = None, weights_backbone = config.model.pretrained_backbone)
    
    model.classifier[4] = torch.nn.Conv2d(256, config.model.n_class, kernel_size=(1,1), stride=(1,1))
    model.backbone.conv1 = nn.Conv2d(config.model.n_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)


    #Load and overwrite model
    saved_model_path = os.path.join(training_path, 'best_model.pth')
    model.load_state_dict(torch.load(saved_model_path))

    #Set weights to 0
    if eval_type == "zero_out":
        with torch.no_grad():
            model.backbone.conv1.weight[:, 3:5, :, :] = 0

    model.to(config.device)

    model.eval()

    if config.loss_function == "CE":
        eval_loss_f = nn.CrossEntropyLoss()

    eval_loss = []
    val_iter = iter(val_loader)
    y_pred_list = []
    y_list = []

    for batch in tqdm(val_iter):
        x, y = batch

        noise_level = 1.0 #want it to be only noise

        if eval_type == "salt_n_pepper":
            for i in range(len(x)):
                x[i,3,:,:] = salt_n_pepper(config, x[i,3, :, :], noise_level, 3)
                x[i,4,:,:] = salt_n_pepper(config, x[i,4, :, :], noise_level, 4)

        if eval_type == "pixel_wise_fade":
            for i in range(len(x)):
                x[i,3,:,:] = pixel_wise_fade(config, x[i,3, :, :], noise_level, 3)
                x[i,4,:,:] = pixel_wise_fade(config, x[i,4, :, :], noise_level, 4)

        if eval_type == "image_wise_fade":
            for i in range(len(x)):
                x[i,3,:,:] = image_wise_fade(config, x[i,3, :, :], noise_level, 3)
                x[i,4,:,:] = image_wise_fade(config, x[i,4, :, :], noise_level, 4)

        x = x.to(config.device)
        y = y.to(config.device)

        with torch.no_grad():
            #y_pred = model(x)['out']
            y_pred = model(x)

        l = eval_loss_f(y_pred, y)
        eval_loss.append(l.item())

        y_pred = torch.argmax(y_pred, dim=1)
        y_pred = y_pred.to(torch.uint8).cpu().contiguous().numpy()
        y = y.to(torch.uint8).cpu().contiguous().numpy()
        y_pred_list.append(y_pred)
        y_list.append(y)

    l_test = np.mean(eval_loss)
   
    writer.add_scalar('evaluation/loss', l_test)
    miou_prec_rec_writing(config, y_pred_list, y_list, 'evaluation', writer, 0)
    miou_prec_rec_writing_13(y_pred_list, y_list, 'evaluation', writer, 0)
    conf_matrix(config, y_pred_list, y_list, writer, 0)

    def salt_n_pepper(config, x_priv_layer, noise_level, channel_num):
        H, W = x_priv_layer.shape
        num_pixels_to_modify = int(noise_level * H * W)

        #create a mask for pixels to modify
        mask = torch.zeros(H * W, dtype=torch.bool)
        mask[:num_pixels_to_modify] = True
        torch.randperm(H * W, out=mask.view(-1)) #Shuffle

        #shape mask back to match image dimensions
        mask = mask.view(H, W)

        #set "black" and "white" values
        black_value = np.array[config.dataset.mean][channel_num] + 5*np.array[config.dataset.std][channel_num]
        white_value = np.array[config.dataset.mean][channel_num] - 5*np.array[config.dataset.std][channel_num]# Randomly choose between black_value and white_value for the selected pixels
        
        #Randomly choose between black_value and white_value for the selected pixels
        black_or_white = torch.rand(H, W) < 0.5
        black_or_white = black_or_white * (white_value - black_value) + black_value

        # Apply the noise to the image
        x_priv_noise = torch.where(mask, black_or_white, x_priv_layer)
        return x_priv_noise
    
    def pixel_wise_fade(config, x_priv_layer, noise_level, channel_num):
        H,W = x_priv_layer.shape
        num_pixels_to_modify = int(noise_level * H * W)

        mask = torch.zeros(H * W, dtype=torch.bool)
        mask[:num_pixels_to_modify] = True
        torch.randperm(H * W, out=mask.view(-1))  # Shuffle the mask
        mask = mask.view(H, W)

        # Generate random values for the selected pixels
        mean = np.array[config.dataset.mean][channel_num]
        std = np.array[config.dataset.std][channel_num]
        random_values = torch.normal(mean, std, size=(num_pixels_to_modify,))

        random_values_image = torch.zeros_like(x_priv_layer)
        random_values_image[mask] = random_values

        # Apply the random values to the image
        x_priv_noise = torch.where(mask, random_values_image, x_priv_layer)

        return x_priv_noise

    def image_wise_fade(config, x_priv_layer, noise_level, channel_num):
        H, W = x_priv_layer
        mean = np.array[config.dataset.mean][channel_num]
        std = np.array[config.dataset.std][channel_num]
        random_image = torch.normal(mean, std, size=(H,W))

        x_priv_noise = (1-noise_level)*x_priv_layer + noise_level*random_image

        return x_priv_noise
    
    def batch_chaos(config, x_priv, noise_level):
        #Use the method that seems to work the best, and do noise based on batch, not relevant here
        return x_priv