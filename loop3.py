import torch
import torch.nn as nn
import numpy as np
import util
from tqdm import tqdm
from torchvision.models.segmentation.deeplabv3 import deeplabv3_resnet50
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
import sys
#from torch.cuda.amp import autocast, GradScaler
from fcnpytorch.fcn8s import FCN8s as FCN8s #smaller net!
import os
import math
import random
import cv2
import matplotlib.pyplot as plt
from dice_loss import DiceLoss


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

    # Set values > 12 to 13
    y_pred_list[y_pred_list > 12] = 13
    y_list[y_list > 12] = 13

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
            #print(x.shape) #np array, shape C,512, 512
            #print(y_pred.shape) #512,512
            #print(y.shape) #512, 512
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

def loop3(config, writer, hydra_log_dir):
    dataset_module = util.load_module(config.dataset.script_location)
    train_set = dataset_module.train_set(config)
    val_set = dataset_module.val_set(config)
    
    #save label image for reference
    label_image(config, writer)
    
    train_loader = DataLoader(train_set, batch_size = config.batch_size, shuffle = True, num_workers = config.num_workers,
                              pin_memory = True)
    val_loader = DataLoader(val_set, batch_size = config.val_batch_size, shuffle = False, num_workers = config.num_workers, 
                            pin_memory = True)
    
    model = deeplabv3_resnet50(weights = config.model.pretrained, progress = True, #num_classes = config.model.n_class,
                                dim_input = config.model.n_channels, aux_loss = None, weights_backbone = config.model.pretrained_backbone)
    
    model.classifier[4] = torch.nn.Conv2d(256, config.model.n_class, kernel_size=(1,1), stride=(1,1))
    model.backbone.conv1 = nn.Conv2d(config.model.n_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    #model = FCN8s(n_class=config.model.n_class, dim_input=config.model.n_channels, weight_init='normal')

    model.to(config.device)
    #scaler = GradScaler()

    if config.optimizer == 'adam':
        #TODO: Introduce config option for betas
        optimizer = Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay, betas=(config.beta1, 0.999))

    if config.optimizer == 'SGD':
        optimizer = SGD(model.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)

    #NOTE: CE loss might not be the best to use for semantic segmentation, look into jaccard losses.
    if config.loss_function =='CE':
        train_loss = nn.CrossEntropyLoss()
        eval_loss = nn.CrossEntropyLoss()   

    if config.loss_function == 'dice':
        train_loss = DiceLoss(config) #dice loss is a modified version of jaccard
        eval_loss =  DiceLoss(config)
    
    epoch = 0
    best_val_loss = np.inf

    while epoch < config.max_epochs:
        print(torch.cuda.current_device())
        print(torch.cuda.is_available())
        print(next(model.parameters()).device)
        
        print('Epoch: '+str(epoch))
        epoch_loss = [] 
        model.train()
        train_iter = iter(train_loader)

        y_pred_list = [] #list to save for an entire epoch
        y_list = []

        print_me = True
        
        for batch in tqdm(train_iter):
            x,y = batch
            x = x.to(config.device, dtype=torch.float32)
            y = y.to(config.device)         
            
            optimizer.zero_grad()
            #with autocast():
            y_pred = model(x)['out'] #NOTE: dlv3_r50 returns a dictionary
            #y_pred.requires_grad()
            #y_pred = torch.argmax(y_pred, dim=1) #sets class to each data point
                #y_pred = model(x)
            l = train_loss(y_pred, y)
            #l.requires_grad(True)
            y_pred = torch.argmax(y_pred, dim=1)

            if print_me:
                print('efter back: ')
                print(l)
                

            l.backward()

            if print_me:
                print('efter back: ')
                print(l)
                print_me = False

            optimizer.step()
            #scaler.scale(l).backward()
            #scaler.step(optimizer)
            #scaler.update()
            

            #y_pred and y has shape: batch_size, crop_size, crop_size, save all values as uint8 in lists on RAM
            y_pred = y_pred.to(torch.uint8).cpu().detach().contiguous().numpy()
            y = y.to(torch.uint8).cpu().detach().contiguous().numpy()
            y_pred_list.append(y_pred)
            y_list.append(y)

            epoch_loss.append(l.item())
            

        #Save loss
        writer.add_scalar('train/loss', np.mean(epoch_loss), epoch)
        print('Epoch mean loss: '+str(np.mean(epoch_loss)))

        #Move model off from VRAM when performing heavy calculations
        miou_prec_rec_writing(config, y_pred_list, y_list, part='train', writer=writer, epoch=epoch)
        miou_prec_rec_writing_13(config, y_pred_list, y_list, part='train', writer=writer, epoch=epoch)

        #clean
        del y_list, y_pred_list

        if epoch % config.eval_every == 0:
            model.eval()
            val_loss = []
            val_y_pred_list = []
            val_y_list = []

            num_batches = math.floor(len(val_set)/config.val_batch_size)
            same_img_idx = 1
            random_img_idx_1 = random.randint(2, math.floor(num_batches/2))
            random_img_idx_2 = random.randint(math.floor(num_batches/2)+1, num_batches-1)
            idx_list = [same_img_idx, random_img_idx_1, random_img_idx_2]
            counter = 0

            val_iter = iter(val_loader)
            for batch in tqdm(val_iter):
                x,y = batch
                x = x.to(config.device)
                y = y.to(config.device)
                y_pred = model(x)['out']
                #y_pred = model(x)
                l = eval_loss(y_pred, y)
                
                y_pred = torch.argmax(y_pred, dim=1)
                val_loss.append(l.item())
        
                if counter in idx_list and epoch % 30 == 0:
                    x_cpu =  x[0, :, :, :].cpu().detach().contiguous().numpy()
                    y_pred_cpu = y_pred[0, :, :].to(torch.uint8).cpu().detach().contiguous().numpy()
                    y_cpu = y[0, :, :].to(torch.uint8).cpu().detach().contiguous().numpy()
                    save_image(counter, x_cpu, y_pred_cpu, y_cpu, epoch, config, writer)

                y_pred = y_pred.to(torch.uint8).cpu().contiguous().detach().numpy()
                y = y.to(torch.uint8).cpu().contiguous().detach().numpy()
                val_y_pred_list.append(y_pred)
                val_y_list.append(y)

                counter += 1

            #Save loss
            l_val = np.mean(val_loss)
            writer.add_scalar('val/loss', l_val, epoch)
            print('Val loss: '+str(l_val))

            miou_prec_rec_writing(config, val_y_pred_list, val_y_list, part='val', writer=writer, epoch=epoch)
            miou_prec_rec_writing_13(config, val_y_pred_list, val_y_list, part='val', writer=writer, epoch=epoch)

            del val_y_list, val_y_pred_list

            if l_val < best_val_loss and epoch != 0:
                best_val_loss = l_val
                torch.save(model.state_dict(), os.path.join(hydra_log_dir, 'best_model.pth'))
                if config.save_optimizer:
                    #will not work
                    torch.save(optimizer.state_dict(), 'best_optimizer.pth')
        
        if epoch % config.save_model_freq == 0 and epoch != 0:
            torch.save(model.state_dict(), os.path.join(hydra_log_dir, 'model_'+str(epoch)+'.pth'))
            if config.save_optimizer:
                #will not work
                torch.save(optimizer.state_dict(), 'optimizer_'+str(epoch)+'.pth')

        epoch +=1

   

    




    


        