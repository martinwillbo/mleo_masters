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

def loop(config, writer = None):

    dataset_module = util.load_module(config.dataset.script_location)
    train_set = dataset_module.train_set(config)
    val_set = dataset_module.val_set(config)
    #NOTE: Just outlining the 'interface' of this way of structuring the experiment code
    #test_set = dataset_module.test_set(config)

    train_loader = DataLoader(train_set, batch_size = config.batch_size, shuffle = True, num_workers = config.num_workers,
                              pin_memory = True)
    val_loader = DataLoader(val_set, batch_size = config.val_batch_size, shuffle = False, num_workers = config.num_workers, 
                            pin_memory = True)
    
    #NOTE: There is a implementation difference here between dataset and model, we could have used the same scheme for the model.
    #Just showcasing two ways of doing things. This approach is 'simpler' but offers less modularity (which is always not bad).
    #If we intend to mainly work with one model and don't need to wrap it in custom code or whatever this is fine.
    #model = deeplabv3_resnet50(weights = config.model.pretrained, progress = True, num_classes = config.model.n_class,
    #                            dim_input = config.model.n_channels, aux_loss = None, weights_backbone = config.model.pretrained_backbone)
    
    model = FCN8s(n_class=config.model.n_class, dim_input=config.model.n_channels, weight_init='normal')
    model.to(config.device)
    num_params = sum(p.numel() for p in model.parameters())
    size_in_bits = num_params * 32/1000000/8
    print(f"Model size: {size_in_bits} MB")
    #add first layer so to have 5 channels, or switch net to one which can take params

    # Initialize GradScaler
    scaler = GradScaler()

    if config.optimizer == 'adam':
        #TODO: Introduce config option for betas
        optimizer = Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay, betas=(config.beta1, 0.999))

    if config.optimizer == 'SGD':
        optimizer = SGD(model.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)

    #WHY WOULD WE SET THE TRANSFORM HERE? Seems more reasonable to set it in __get_item__
    #if config.use_transform:
    #    transform_module = util.load_module(config.transform.script_location)
    #    transform = transform_module.get_transform(config)
    #    train_set.set_transform(transform)

    #NOTE: CE loss might not be the best to use for semantic segmentation, look into jaccard losses.
    train_loss = nn.CrossEntropyLoss()
    eval_loss = nn.CrossEntropyLoss()
    
    epoch = 0
    best_val_loss = np.inf

    #NOTE: Can also include a check for early stopping counter here.
    while epoch < config.max_epochs:
        print(torch.cuda.current_device())
        print(torch.cuda.is_available())
        print(next(model.parameters()).device)
        
        print('Epoch: '+str(epoch))
        epoch_loss = []
        epoch_miou_prec_rec = np.nan * np.empty((3, config.model.n_class)) #creates empty vecn 
        model.train()
        train_iter = iter(train_loader)

        y_pred_list = [] #list to save for an entire epoch
        y_list = []
        
        for batch in tqdm(train_iter):
            #optimizer.zero_grad()
            x, y = batch
            x = x.to(config.device)
            y = y.to(config.device)
            #NOTE: dlv3_r50 returns a dictionary
            with autocast():
                #y_pred = model(x)['out']
                y_pred = model(x)
                l = train_loss(y_pred, y)
            #y_pred = model(x)['out']
            #print("Calculating loss")
            #l = train_loss(y_pred, y)
            optimizer.zero_grad()
            scaler.scale(l).backward()
            #l.backward()
            scaler.step(optimizer)
            scaler.update()

            #optimizer.step()
            #NOTE: If you have a learning rate scheduler this is to place to step it. 
            y_pred = torch.argmax(y_pred, dim=1) #sets class to each data point
            #y_pred and y has shape: batch_size, crop_size, crop_size
            #save all values as uint8 in lists  
            y_pred = y_pred.to(torch.uint8).cpu().contiguous().numpy()
            y = y.to(torch.uint8).cpu().contiguous().numpy()
            y_pred_list.append(y_pred)
            y_list.append(y)

            #y_pred = y_pred.cpu().contiguous()
            #y = y.cpu().contiguous()
            #y_pred_flat = y_pred.view(-1).numpy() 
            #y_flat = y.view(-1).numpy()
            #iou_prec_rec = np.nan * np.empty((3, config.model.n_class)) #creates empty vec
            #for i in range(config.model.n_class): #for all classes
            #    y_flat_i = y_flat == i #sets ones where y_flat is equal to i
            #    num_i = np.count_nonzero(y_flat_i) #count nbr of occurances of class i in true y
            #    pred_flat_i = y_pred_flat == i 
            #    num_pred_i = np.count_nonzero(pred_flat_i)
            #    intersection_i = np.logical_and(y_flat_i, pred_flat_i) #where they match
            #    union_i = np.logical_or(y_flat_i, pred_flat_i) #everything together
            #    num_intersection_i = np.count_nonzero(intersection_i) #how big is the intersection
            #    num_union_i = np.count_nonzero(union_i) #how big is the union
            #    if num_union_i > 0: 
            #        iou_prec_rec[0,i] = num_intersection_i/num_union_i
            #    if num_pred_i > 0:
            #        iou_prec_rec[1,i] = num_intersection_i / num_pred_i
            #    if num_i > 0:
            #        iou_prec_rec[2,i] = num_intersection_i / num_i
            #epoch_miou_prec_rec.append(iou_prec_rec)
            epoch_loss.append(l.item())
        
        #FIXED
        y_pred_list = np.concatenate(y_pred_list, axis=0)
        y_list = np.concatenate(y_list, axis=0)
        #y_pred_list = y_pred_list.cpu().contiguous()
        #y_list = y_list.cpu().contiguous()  
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
        #FIXED

        #epoch_miou_prec_rec = np.nanmean(np.stack(epoch_miou_prec_rec, axis = 0), axis = 0)
        #print(epoch_miou_prec_rec)
        epoch_miou_prec_rec = np.nan_to_num(epoch_miou_prec_rec,nan=0.0) #set nans to 0, bc we are not predicting even when we should
        writer.add_scalar('train/loss', np.mean(epoch_loss), epoch)
        print('Epoch mean loss: '+str(np.mean(epoch_loss)))
        writer.add_scalar('train/miou', np.mean(epoch_miou_prec_rec[0,:]), epoch)
        writer.add_scalar('train/precision', np.mean(epoch_miou_prec_rec[1,:]), epoch)
        writer.add_scalar('train/recall', np.mean(epoch_miou_prec_rec[2,:]), epoch)
        #First 12 classes only for ref
        writer.add_scalar('train/miou first 12', np.mean(epoch_miou_prec_rec[0,:12]), epoch)
        writer.add_scalar('train/precision first 12', np.mean(epoch_miou_prec_rec[1,:12]), epoch)
        writer.add_scalar('train/recall first 12', np.mean(epoch_miou_prec_rec[2,:12]), epoch)
        #Also add class specific values
        writer.add_text('train/miou per class', ', '.join(map(str, epoch_miou_prec_rec[0,:])), epoch)
        writer.add_text('train/precision per class', ', '.join(map(str, epoch_miou_prec_rec[1,:])), epoch)
        writer.add_text('train/recall per class', ', '.join(map(str, epoch_miou_prec_rec[2,:])), epoch)
        print('Epoch mean miou: '+str(np.mean(epoch_miou_prec_rec[0,:])))
        print('Epoch mean precision: '+str(np.mean(epoch_miou_prec_rec[1,:])))
        print('Epoch mean recall: '+str(np.mean(epoch_miou_prec_rec[2,:])))

        if epoch % config.eval_every == 0:
            #NOTE: I added some basic eval code here. Typically I hide it in a module, depends on how much it is.
            model.eval()
            val_loss = []
            val_miou_prec_rec = np.nan * np.empty((3, config.model.n_class))

            val_y_pred_list = []
            val_y_list = []

            val_iter = iter(val_loader)
            for batch in tqdm(val_iter):
                x, y = batch
                x = x.to(config.device)
                y = y.to(config.device)
                #y_pred = model(x)['out']
                y_pred = model(x)
                y_pred = torch.argmax(y_pred, dim=1)

                y_pred = y_pred.to(torch.uint8).cpu().contiguous().numpy()
                y = y.to(torch.uint8).cpu().contiguous().numpy()
                val_y_pred_list.append(y_pred)
                val_y_list.append(y)


                #y_pred = y_pred.cpu().contiguous()
                #y = y.cpu().contiguous()
                #y_pred_flat = y_pred.view(-1).numpy()
                #y_flat = y.view(-1).numpy()
                #iou_prec_rec = np.nan * np.empty((3, config.model.n_class))
                #for i in range(config.model.n_class):
                #    y_flat_i = y_flat == i
                #    num_i = np.count_nonzero(y_flat_i)
                #    pred_flat_i = y_pred_flat == i
                #    num_pred_i = np.count_nonzero(pred_flat_i)
                #    intersection_i = np.logical_and(y_flat_i, pred_flat_i)
                #    union_i = np.logical_or(y_flat_i, pred_flat_i)
                #    num_intersection_i = np.count_nonzero(intersection_i)
                #    num_union_i = np.count_nonzero(union_i)
                #    if num_union_i > 0:
                #        iou_prec_rec[0,i] = num_intersection_i/num_union_i
                #    if num_pred_i > 0:
                #        iou_prec_rec[1,i] = num_intersection_i / num_pred_i
                #    if num_i > 0:
                #        iou_prec_rec[2,i] = num_intersection_i / num_i
                #val_miou_prec_rec.append(iou_prec_rec)
                val_loss.append(l.item()) #is this really the correct loss, shouldn't we calc l_val

            #FIXED
            val_y_pred_list = np.concatenate(val_y_pred_list, axis=0)
            val_y_list = np.concatenate(val_y_list, axis=0)  
            val_y_pred_flat_list = val_y_pred_list.reshape(-1)
            val_y_flat_list = val_y_list.reshape(-1)
        
            for i in range(config.model.n_class): #for all classes
                y_flat_i = val_y_flat_list == i #sets ones where y_flat is equal to i
                num_i = np.count_nonzero(y_flat_i) #count nbr of occurances of class i in true y
                pred_flat_i = val_y_pred_flat_list == i 
                num_pred_i = np.count_nonzero(pred_flat_i)
                intersection_i = np.logical_and(y_flat_i, pred_flat_i) #where they match
                union_i = np.logical_or(y_flat_i, pred_flat_i) #everything together
                num_intersection_i = np.count_nonzero(intersection_i) #how big is the intersection
                num_union_i = np.count_nonzero(union_i) #how big is the union
                if num_union_i > 0: 
                    val_miou_prec_rec[0,i] = num_intersection_i/num_union_i
                if num_pred_i > 0:
                    val_miou_prec_rec[1,i] = num_intersection_i / num_pred_i
                if num_i > 0:
                    val_miou_prec_rec[2,i] = num_intersection_i / num_i
            #FIXED

            #val_miou_prec_rec = np.nanmean(np.stack(val_miou_prec_rec, axis = 0), axis = 0)
            l_val = np.mean(val_loss)
            writer.add_scalar('val/loss', l_val, epoch)
            val_miou_prec_rec = np.nan_to_num(val_miou_prec_rec,nan=0.0)
            writer.add_scalar('val/miou', np.mean(val_miou_prec_rec[0,:]), epoch)
            writer.add_scalar('val/precision', np.mean(val_miou_prec_rec[1,:]), epoch)
            writer.add_scalar('val/recall', np.mean(val_miou_prec_rec[2,:]), epoch)
            #First 12 classes only for ref
            writer.add_scalar('val/miou first 12', np.mean(val_miou_prec_rec[0,:12]), epoch)
            writer.add_scalar('val/precision first 12', np.mean(val_miou_prec_rec[1,:12]), epoch)
            writer.add_scalar('val/recall first 12', np.mean(val_miou_prec_rec[2,:12]), epoch)
            #Save per class
            writer.add_text('val/miou per class', ', '.join(map(str, val_miou_prec_rec[0,:])), epoch)
            writer.add_text('val/precision per class', ', '.join(map(str, val_miou_prec_rec[1,:])), epoch)
            writer.add_text('val/recall per class', ', '.join(map(str, val_miou_prec_rec[2,:])), epoch)
            print('Val loss: '+str(l_val))
            print('Val mean miou: '+str(np.mean(val_miou_prec_rec[0,:])))
            print('Val mean precision: '+str(np.mean(val_miou_prec_rec[1,:])))
            print('Val mean recall: '+str(np.mean(val_miou_prec_rec[2,:])))
            if l_val < best_val_loss:
                best_val_loss = l_val
                torch.save(model.state_dict(), 'best_model.pth')
                if config.save_optimizer:
                    torch.save(optimizer.state_dict(), 'best_optimizer.pth')

        if epoch % config.save_model_freq == 0:
            torch.save(model.state_dict(), 'model_'+str(epoch)+'.pth')
            if config.save_optimizer:
                torch.save(optimizer.state_dict(), 'optimizer_'+str(epoch)+'.pth')

        epoch += 1