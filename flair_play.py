# Imports
import os, sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy as np
import tifffile as tiff
from PIL import Image
#import torch

# Global vars
BASE_PATH_DATA = '/raid/aleksispi/master-theses/agnes-malte-spring2024/flair_2_toy_dataset'
SPLIT_TO_USE = 'train' #Which split to use
AERIAL_BASE_PATH = os.path.join(BASE_PATH_DATA, 'flair_2_toy_aerial_' + SPLIT_TO_USE)
LABEL_BASE_PATH = os.path.join(BASE_PATH_DATA, 'flair_2_toy_labels_' + SPLIT_TO_USE)
SENTINEL_BASE_PATH = os.path.join(BASE_PATH_DATA, 'flair_2_toy_sen_' + SPLIT_TO_USE)

# Create list of all full paths to tif files within (subfolders of)
# AERIAL_BASE_PATH
aerial_tif_paths = [] #vector of all aearial photos
for root, dirs, files in os.walk(AERIAL_BASE_PATH):
    for file in files:
        if file.endswith(".tif"):
             aerial_tif_paths.append(os.path.join(root, file))

# As above but for label tif files
label_tif_paths = []
for root, dirs, files in os.walk(LABEL_BASE_PATH):
    for file in files:
        if file.endswith(".tif"):
             label_tif_paths.append(os.path.join(root, file))

# As above but for sentinel npy files
sentinel_data_paths = []
sentinel_mask_paths = []
for root, dirs, files in os.walk(SENTINEL_BASE_PATH):
    for file in files:
        if file.endswith("_data.npy"):
             sentinel_data_paths.append(os.path.join(root, file))
        elif file.endswith("_masks.npy"):
            sentinel_mask_paths.append(os.path.join(root, file))
#print(len(aerial_tif_paths))
#print(len(label_tif_paths))
#print(len(sentinel_data_paths))
#print(len(sentinel_mask_paths))
# Assert that the number of aerial and sentinel tifs are the same
assert len(aerial_tif_paths) == len(label_tif_paths) == len(sentinel_data_paths) == len(sentinel_mask_paths)

# Assert that all folder lists (aerial_tif_paths, label_tif_paths, sentinel_data_paths,
# sentinel_mask_paths) are in the same order
for i in range(len(aerial_tif_paths)):
    aerial_base = ""
    for component in aerial_tif_paths[i].split('/')[-4:-2]:
        aerial_base = os.path.join(aerial_base, component)
    label_base = ""
    for component in label_tif_paths[i].split('/')[-4:-2]:
        label_base = os.path.join(label_base, component)
    sentinel_data_base = ""
    for component in sentinel_data_paths[i].split('/')[-4:-2]:
        sentinel_data_base = os.path.join(sentinel_data_base, component)
    sentinel_mask_base = ""
    for component in sentinel_mask_paths[i].split('/')[-4:-2]:
        sentinel_mask_base = os.path.join(sentinel_mask_base, component)
    assert aerial_base == label_base == sentinel_data_base == sentinel_mask_base

def plot_data(path_idx=0):
    # Use matplotlib to plot the first aerial and label tif side-by-side and
    # and on the second row also plot the first sentinel data and mask npy files,
    # then save the result to a file
    fig = plt.figure(figsize=(16, 16))
    # Note that it seems like the first 3 channels are RGB
    # Based on the figure at https://github.com/IGNF/FLAIR-2-AI-Challenge,
    # it seems as if the 4th channel is NIR and the 5th is Elevations
    aerial_img = tiff.imread(aerial_tif_paths[path_idx])
    fig.add_subplot(2,2,1)
    plt.imshow(aerial_img[:, :, :3])
    plt.title('Aerial RGB')
    label_img = tiff.imread(label_tif_paths[path_idx])

    # Show the label image as a color image
    fig.add_subplot(2,2,2)
    plt.imshow(label_img)
    plt.title('Label image')

    if False:
        # Show the sentinel data and mask as a color image
        sentinel_data = np.load(sentinel_data_paths[path_idx])
        sentinel_mask = np.load(sentinel_mask_paths[path_idx])
        print(aerial_img.shape, label_img.shape)
        print(sentinel_data.shape, sentinel_mask.shape)
        fig.add_subplot(2,2,3)
        plt.imshow(sentinel_data)
        plt.title('Sentinel Data')
        fig.add_subplot(2,2,4)
        plt.imshow(sentinel_mask)
    else:
        # In this case instead show the NIR and elevation channels
        # For the NIR, use R-G coloring
        # For the elevation, use a grayscale colormap
        fig.add_subplot(2,2,3)
        # TODO: Figure out how to do the R-G coloring
        plt.imshow(aerial_img[:, :, 3])
        plt.title('Aerial NIR')
        fig.add_subplot(2,2,4)
        plt.imshow(aerial_img[:, :, 4], cmap='gray')
        plt.title('Aerial Elevation')

    # Save the figure
    plt.savefig('aerial_label_%d.png' % path_idx)
    plt.cla()
    plt.clf()
    plt.close('all')

if True:
    for i in range(5):
        print(label_tif_paths)
        plot_data(i)

# Sanity check (the below code simply to check that GPU-cuda stuff works)
# Transform the loaded aerial npy into torch tensor and send to GPU
aerial_img = tiff.imread(aerial_tif_paths[0])
aerial_img = aerial_img[:, :, :3]
aerial_img = torch.from_numpy(aerial_img).float()
aerial_img = aerial_img.to('cuda')

print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())
