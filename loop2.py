import torch
import torch.nn as nn
import numpy as np
import util
from tqdm import tqdm
from torchvision.models.segmentation.deeplabv3 import deeplabv3_resnet50
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
import sys
from torch.cuda.amp import autocast, GradScaler
from fcnpytorch.fcn8s import FCN8s as FCN8s #smaller net!


def miou_prec_rec_writing(config, y_pred_list, y_list, part, writer, epoch):
    epoch_miou_prec_rec = np.nan * np.empty((3, config.model.n_class)) #creates empty vecn
    y_pred_list = np.concatenate(y_pred_list, axis=0)
    y_list = np.concatenate(y_list, axis=0)
    y_pred_flat_list = y_pred_list.reshape(-1)
    y_flat_list = y_list.reshape(-1)

    for i in range(config.model.n_class): #for all classes
        y_flat_i = y_flat_list == i #sets ones where y_flat is equal to i
        num_i = np.count_nonzero(y_flat_i) #count nbr of occurances of class i in true y
        pred_flat_i = y_pred_flat_list == i 
        num_pred_i = np.count_nonzero(pred_flat_i)
        intersection_i = np.logical_and(y_flat_i, pred_flat_i) #where they match
        union_i = np.logical_or(y_flat_i, pred_flat_i) #everything together
        num_intersection_i = np.count_nonzero(intersection_i) #how big is the intersection
        num_union_i = np.count_nonzero(union_i) #how big is the union
        if num_union_i > 0: 
            epoch_miou_prec_rec[0,i] = num_intersection_i/num_union_i
        if num_pred_i > 0:
            epoch_miou_prec_rec[1,i] = num_intersection_i / num_pred_i
        if num_i > 0:
            epoch_miou_prec_rec[2,i] = num_intersection_i / num_i

    epoch_miou_prec_rec = np.nan_to_num(epoch_miou_prec_rec,nan=0.0) #set nans to 0, bc we are not predicting even when we should

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

def miou_prec_rec_writing_13(y_pred_list, y_list, part, writer, epoch):
        epoch_miou_prec_rec = np.nan * np.empty((3, 1)) #creates empty vecn
        y_pred_list = np.concatenate(y_pred_list, axis=0)
        y_list = np.concatenate(y_list, axis=0)
        y_pred_flat_list = y_pred_list.reshape(-1)
        y_flat_list = y_list.reshape(-1)
        y_pred_flat_list[y_pred_flat_list > 12] = 13
        y_flat_list[y_flat_list > 12] = 13    

        y_flat_13 = y_flat_list == 13 #sets ones where y_flat is equal to i
        num_13 = np.count_nonzero(y_flat_13) #count nbr of occurances of class i in true y
        pred_flat_13 = y_pred_flat_list == 13 
        num_pred_13 = np.count_nonzero(pred_flat_13)
        intersection_13 = np.logical_and(y_flat_13, pred_flat_13) #where they match
        union_13 = np.logical_or(y_flat_13, pred_flat_13) #everything together
        num_intersection_13 = np.count_nonzero(intersection_13) #how big is the intersection
        num_union_13 = np.count_nonzero(union_13) #how big is the union
        if num_union_13 > 0: 
            epoch_miou_prec_rec[0,0] = num_intersection_13/num_union_13
        if num_pred_13 > 0:
            epoch_miou_prec_rec[1,0] = num_intersection_13 / num_pred_13
        if num_13 > 0:
            epoch_miou_prec_rec[2,0] = num_intersection_13 / num_13

        #Save into writer
        writer.add_scalar(part+'/miou fixed 13th class', epoch_miou_prec_rec[0,0], epoch)
        writer.add_scalar(part+'/precision fixed 13th class', epoch_miou_prec_rec[1,0], epoch)
        writer.add_scalar(part+'/recall fixed 13th class', epoch_miou_prec_rec[2,0], epoch)

def loop2(config, writer=None):
    dataset_module = util.load_module(config.dataset.script_location)
    train_set = dataset_module.train_set(config)
    val_set = dataset_module.val_set(config)

    
    train_loader = DataLoader(train_set, batch_size = config.batch_size, shuffle = False, num_workers = config.num_workers,
                              pin_memory = True)
    val_loader = DataLoader(val_set, batch_size = config.val_batch_size, shuffle = False, num_workers = config.num_workers, 
                            pin_memory = True)
    
    #model = deeplabv3_resnet50(weights = config.model.pretrained, progress = True, num_classes = config.model.n_class,
    #                            dim_input = config.model.n_channels, aux_loss = None, weights_backbone = config.model.pretrained_backbone)
    
    model = FCN8s(n_class=config.model.n_class, dim_input=config.model.n_channels, weight_init='normal')
    model.to(config.device)

    scaler = GradScaler()

    if config.optimizer == 'adam':
        #TODO: Introduce config option for betas
        optimizer = Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay, betas=(config.beta1, 0.999))

    if config.optimizer == 'SGD':
        optimizer = SGD(model.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)

    #NOTE: CE loss might not be the best to use for semantic segmentation, look into jaccard losses.
    train_loss = nn.CrossEntropyLoss()
    eval_loss = nn.CrossEntropyLoss()
    
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

        for batch in tqdm(train_iter):
            x,y = batch
            x = x.to(config.device)
            y = y.to(config.device)
            
            with autocast():
                #y_pred = model(x)['out'] #NOTE: dlv3_r50 returns a dictionary
                y_pred = model(x)
                l = train_loss(y_pred, y)
            optimizer.zero_grad()
            scaler.scale(l).backward()
            scaler.step(optimizer)
            scaler.update()

            y_pred = torch.argmax(y_pred, dim=1) #sets class to each data point
            #y_pred and y has shape: batch_size, crop_size, crop_size, save all values as uint8 in lists on RAM
            y_pred = y_pred.to(torch.uint8).cpu().contiguous().numpy()
            y = y.to(torch.uint8).cpu().contiguous().numpy()
            y_pred_list.append(y_pred)
            y_list.append(y)

            epoch_loss.append(l.item())

        #Save loss
        writer.add_scalar('train/loss', np.mean(epoch_loss), epoch)
        print('Epoch mean loss: '+str(np.mean(epoch_loss)))

        miou_prec_rec_writing(config, y_pred_list, y_list, part='train', writer=writer, epoch=epoch)
        miou_prec_rec_writing_13(y_pred_list, y_list, part='train', writer=writer, epoch=epoch)

        if epoch % config.eval_every == 0:
            model.eval()
            val_loss = []
            val_y_pred_list = []
            val_y_list = []

            val_iter = iter(val_loader)
            for batch in tqdm(val_iter):
                x,y = batch
                x = x.to(config.device)
                y = y.to(config.device)
                #y_pred = model(x)['out']
                y_pred = model(x)
                y_pred = torch.argmax(y_pred, dim=1)

                y_pred = y_pred.to(torch.uint8).cpu().contiguous().numpy()
                y = y.to(torch.uint8).cpu().contiguous().numpy()
                val_y_pred_list.append(y_pred)
                val_y_list.append(y)

                val_loss.append(l.item())

            #Save loss
            l_val = np.mean(val_loss)
            writer.add_scalar('val/loss', l_val, epoch)
            print('Val loss: '+str(l_val))

            miou_prec_rec_writing(config, val_y_pred_list, val_y_list, part='val', writer=writer, epoch=epoch)
            miou_prec_rec_writing_13(val_y_pred_list, val_y_list, part='val', writer=writer, epoch=epoch)

            if l_val < best_val_loss:
                best_val_loss = l_val
                torch.save(model.state_dict(), 'best_model.pth')
                if config.save_optimizer:
                    torch.save(optimizer.state_dict(), 'best_optimizer.pth')
        
        if epoch % config.save_model_freq == 0:
            torch.save(model.state_dict(), 'model_'+str(epoch)+'.pth')
            if config.save_optimizer:
                torch.save(optimizer.state_dict(), 'optimizer_'+str(epoch)+'.pth')

        epoch +=1

   

    




    


        