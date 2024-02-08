import torch
import torch.nn as nn
import numpy as np
import util
from tqdm import tqdm
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from io import BytesIO
from PIL import Image
import torchvision.transforms as transforms


class_names = [
    "building",
    "pervious surface",
    "impervious surface",
    "bare soil",
    "water",
    "coniferous",
    "deciduous",
    "brushwood",
    "vineyard",
    "herbaceous vegetation",
    "agricultural land",
    "plowed land",
    "swimming pool",
    "snow",
    "clear cut",
    "mixed",
    "ligneous",
    "greenhouse",
    "other"
]

colormap = [
    [255, 0, 255],    # Magenta
    [128, 128, 128],  # Grey
    [255, 0, 0],      # Red
    [139, 69, 19],    # Brown
    [0, 0, 255],      # Blue
    [0, 100, 0],      # Dark Green
    [0, 255, 155],    # Greenblue
    [255, 165, 0],    # Orange
    [128, 0, 128],    # Purple
    [0, 220, 0],      # Green
    [255, 255, 0],    # Yellow
    [220, 245, 220],  # Beige
    [64, 224, 228],   # Turquoise
    [255, 255, 255],  # White
    [100, 190, 160],  # Grey Green
    [52, 61, 15],   # Brown Green
    [200, 200, 50],   # Yellow Green
    [190, 170, 220],  # Light Purple
    [0, 0, 0]         # Black
]

def miou_prec_rec_writing(config, y_pred_list, y_list, part, writer, epoch):
    y_pred_list = torch.tensor(np.concatenate(y_pred_list, axis=0))
    y_list = torch.tensor(np.concatenate(y_list, axis=0))

    y_pred_list = y_pred_list.view(-1)
    y_list = y_list.view(-1)

    #print(y_pred_list.element_size() * y_pred_list.numel()/1000000) 
    #print(y_list.element_size()*y_list.numel()/1000000)

    #for val: should be 13000*512*512*2 for both, seems correct, and then doubling that -> req 12GB-> too much

    epoch_miou_prec_rec = torch.full((3, config.model.n_class), float('nan'))

    for i in range(config.model.n_class):
        y_flat_i = (y_list == i)
        num_i = torch.count_nonzero(y_flat_i)
        pred_flat_i = (y_pred_list == i)
        num_pred_i = torch.count_nonzero(pred_flat_i)
        intersection_i = torch.logical_and(y_flat_i, pred_flat_i)
        union_i = torch.logical_or(y_flat_i, pred_flat_i)
        num_intersection_i = torch.count_nonzero(intersection_i)
        num_union_i = torch.count_nonzero(union_i)

        if num_union_i > 0:
            epoch_miou_prec_rec[0, i] = num_intersection_i.float() / num_union_i.float()
        if num_pred_i > 0:
            epoch_miou_prec_rec[1, i] = num_intersection_i.float() / num_pred_i.float()
        if num_i > 0:
            epoch_miou_prec_rec[2, i] = num_intersection_i.float() / num_i.float()

    del y_pred_list, y_list

    epoch_miou_prec_rec = torch.nan_to_num(epoch_miou_prec_rec, nan=0.0)
    epoch_miou_prec_rec = epoch_miou_prec_rec.cpu().numpy()
    #Results as mean over all
    writer.add_scalar(part+'/miou', np.mean(epoch_miou_prec_rec[0,:]), epoch)
    writer.add_scalar(part+'/precision', np.mean(epoch_miou_prec_rec[1,:]), epoch)
    writer.add_scalar(part+'/recall', np.mean(epoch_miou_prec_rec[2,:]), epoch)
    #First 12 classes only for ref
    writer.add_scalar(part+'/miou first 12', np.mean(epoch_miou_prec_rec[0,:12]), epoch)
    writer.add_scalar(part+'/precision first 12', np.mean(epoch_miou_prec_rec[1,:12]), epoch)
    writer.add_scalar(part+'/recall first 12', np.mean(epoch_miou_prec_rec[2,:12]), epoch)
    #Also add class specific values
    writer.add_text(part+'/miou per class', ', '.join(map(str, epoch_miou_prec_rec[0,:])), epoch)
    writer.add_text(part+'/precision per class', ', '.join(map(str, epoch_miou_prec_rec[1,:])), epoch)
    writer.add_text(part+'/recall per class', ', '.join(map(str, epoch_miou_prec_rec[2,:])), epoch)
    print('Epoch mean miou: '+str(np.mean(epoch_miou_prec_rec[0,:])))
    print('Epoch mean precision: '+str(np.mean(epoch_miou_prec_rec[1,:])))
    print('Epoch mean recall: '+str(np.mean(epoch_miou_prec_rec[2,:])))

def miou_prec_rec_writing_13(config, y_pred_list, y_list, part, writer, epoch):
    y_pred_list = torch.tensor(np.concatenate(y_pred_list, axis=0), dtype=torch.uint8)
    y_list = torch.tensor(np.concatenate(y_list, axis=0), dtype=torch.uint8)

    # Flatten tensors
    y_pred_list = y_pred_list.view(-1)
    y_list = y_list.view(-1)
    #print(y_pred_list.shape)

    # Set values > 11 to 13
    y_pred_list[y_pred_list > 11] = 13
    y_list[y_list > 11] = 13

    # Create an empty tensor for epoch_miou_prec_rec
    epoch_miou_prec_rec = torch.full((3, 1), float('nan'))

    # Calculate for class 13
    y_flat_13 = y_list == 13
    num_13 = torch.count_nonzero(y_flat_13)
    pred_flat_13 = y_pred_list == 13
    num_pred_13 = torch.count_nonzero(pred_flat_13)
    intersection_13 = torch.logical_and(y_flat_13, pred_flat_13)
    union_13 = torch.logical_or(y_flat_13, pred_flat_13)
    num_intersection_13 = torch.count_nonzero(intersection_13)
    num_union_13 = torch.count_nonzero(union_13)

    del y_list, y_pred_list

    if num_union_13 > 0:
        epoch_miou_prec_rec[0, 0] = num_intersection_13.float() / num_union_13.float()
    if num_pred_13 > 0:
        epoch_miou_prec_rec[1, 0] = num_intersection_13.float() / num_pred_13.float()
    if num_13 > 0:
        epoch_miou_prec_rec[2, 0] = num_intersection_13.float() / num_13.float()

    # Save into writer
    writer.add_scalar(part+'/miou fixed 13th class', epoch_miou_prec_rec[0, 0].item(), epoch)
    writer.add_scalar(part+'/precision fixed 13th class', epoch_miou_prec_rec[1, 0].item(), epoch)
    writer.add_scalar(part+'/recall fixed 13th class', epoch_miou_prec_rec[2, 0].item(), epoch)

def conf_matrix(config, y_pred_list, y_list, writer, epoch):
    y_pred_list = torch.tensor(np.concatenate(y_pred_list, axis=0))
    y_list = torch.tensor(np.concatenate(y_list, axis=0))

    y_pred_list = y_pred_list.view(-1)
    y_list = y_list.view(-1)

    cm = confusion_matrix(y_list.cpu().numpy(), y_pred_list.cpu().numpy())
    cm = cm.astype('float') / cm.sum(axis=1)[:,np.newaxis]


    # Create a confusion matrix plot
    fig, ax = plt.subplots(figsize=(10, 10))
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    plt.colorbar(cax)

    # Set axis labels to class names
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90)
    ax.set_yticklabels(class_names)

    # Display the numbers inside the matrix
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            formatted_value = '{:.2f}'.format(value)
            ax.text(x=j, y=i, s=formatted_value, va='center', ha='center', color='white' if value > cm.max()/2 else 'black')

    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')

    # Save the plot to a buffer
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)

     # Convert buffer to PIL Image
    image = Image.open(buf)

    # Convert PIL Image to tensor
    transform = transforms.ToTensor()
    cm_image = transform(image)

    # Add confusion matrix image to TensorBoard
    writer.add_image('epoch: ' +str(epoch) +' Val/confusion_matrix', cm_image, epoch)

    buf.close()
    plt.close(fig)


def label_image(config, writer):
    legend_image = np.zeros((config.model.n_class * 30, 100, 3), dtype=np.uint8)

    # Populate the legend image with class labels and corresponding colors
    for class_label in range(config.model.n_class):
        legend_image[class_label * 30:(class_label + 1) * 30, :, :] = colormap[class_label]

    # Create class labels as text and overlay them on the legend
    class_labels = [f'Class {i}' for i in range(config.model.n_class)]
    for i, label in enumerate(class_labels):
        plt.text(120, i * 30 + 15, label, fontsize=12, color='white', va='center')

    # Save the legend image
    plt.imsave('legend.png', legend_image)

    # Load the saved legend image and convert it to a tensor
    legend_image = plt.imread('legend.png')
    legend_tensor = torch.from_numpy(legend_image.transpose(2, 0, 1))

    # Add the legend image to TensorBoard
    writer.add_image('Legend Image', legend_tensor, 0)
    writer.add_text("Class Names", "\n".join(class_names), 0)

def save_image(index, x, y_pred, y, epoch, config, writer):
            

            #Unnormalize x and divide by 255 to get range [0,1]

            x_temp = np.transpose(x[:3], (1,2,0)).astype(float)
            x_temp *= np.array(config.dataset.std)[:3]
            x_temp += np.array(config.dataset.mean)[:3]
            x_temp = np.floor(x_temp)
            x_temp = cv2.cvtColor(x_temp.astype(np.uint8), cv2.COLOR_BGR2RGB) #convert from BGR to RGB
            x_temp = np.transpose(x_temp, (2,0,1))
            x[:3] = x_temp/255.0

            colored_y = np.zeros((3, y.shape[0], y.shape[1]), dtype=np.uint8)
            colored_y_pred = np.zeros((3, y_pred.shape[0], y_pred.shape[1]), dtype=np.uint8)

            # Iterate over each class and color the masks
            for class_label in range(config.model.n_class):
                # Create boolean masks for the current class
                class_mask_y = (y == class_label)
                class_mask_y_pred = (y_pred == class_label)

                # Color the masks
                for c in range(3):  # Iterate over color channels (R, G, B)
                    colored_y[c][class_mask_y] = colormap[class_label][c]
                    colored_y_pred[c][class_mask_y_pred] = colormap[class_label][c]

            x_tensor = torch.from_numpy(x)
            
            if config.model.n_channels == 5:
                x_priv = x_tensor[3:]
                #Not ideal normalization to be honest, as we don't know how strong the signal is
                min_vals = x_priv.view(x_priv.size(0), -1).min(dim=1)[0].unsqueeze(-1).unsqueeze(-1)
                max_vals = x_priv.view(x_priv.size(0), -1).max(dim=1)[0].unsqueeze(-1).unsqueeze(-1)
                x_priv = (x_priv - min_vals) / (max_vals - min_vals)
                x_tensor = x_tensor[:3]
                #Have to normalize x and x_priv to [0,1]
                writer.add_image('Epoch: ' + str(epoch) + ', Val/IR priv info, batch: ' + str(index), x_priv[0,:,:].unsqueeze(0), epoch)
                writer.add_image('Epoch: ' + str(epoch) + ', Val/height map priv info, batch: ' + str(index), x_priv[1,:,:].unsqueeze(0), epoch)
            
            colored_y_tensor = torch.from_numpy(colored_y)
            colored_y_pred_tensor = torch.from_numpy(colored_y_pred)
            colored_y_tensor = colored_y_tensor/255.0 
            colored_y_pred_tensor = colored_y_pred_tensor/255.0
            #print('shapes')
            #print(x_tensor.shape)
            #print(colored_y_tensor.shape)
            #print(colored_y_pred_tensor.shape)
            #print(type(x_tensor), x_tensor.dtype)
            #print(type(colored_y_tensor), colored_y_tensor.dtype)
            #print(type(colored_y_pred_tensor), colored_y_tensor.dtype)
            #print(f"colored_y_tensor: max={colored_y_tensor.max().item()}, min={colored_y_tensor.min().item()}")
            #print(f"colored_y_pred_tensor: max={colored_y_pred_tensor.max().item()}, min={colored_y_pred_tensor.min().item()}")

            
            writer.add_image('Epoch: ' + str(epoch) + ', Val/x, batch: ' + str(index), x_tensor, epoch)
            writer.add_image('Epoch: ' + str(epoch) + ', Val/y, batch: ' + str(index), colored_y_tensor, epoch) #unsqueeze adds dim
            writer.add_image('Epoch: ' + str(epoch) + ', Val/y_pred, batch: ' + str(index), colored_y_pred_tensor, epoch)