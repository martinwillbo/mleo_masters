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
import segmentation_models_pytorch as smp
from unet_module import UnetFeatureSenti, UnetSentiUnet, UnetSentiDoubleLoss, UnetSentiUTAE
from loss_functions import senti_loss, teacher_student_loss
import os
import math
import random
import cv2
import matplotlib.pyplot as plt
from support_functions_logging import miou_prec_rec_writing, miou_prec_rec_writing_13, conf_matrix, label_image, save_image, save_senti_image
from support_functions import set_model, teacher_student, get_teacher, collate_fn

def loop3(config, writer, hydra_log_dir):

    dataset_module = util.load_module(config.dataset.script_location)
    train_set = dataset_module.train_set(config)
    val_set = dataset_module.val_set(config)
    
    #save label image for reference
    label_image(config, writer)
    
    train_loader = DataLoader(train_set, batch_size = config.batch_size, shuffle = True, num_workers = config.num_workers,
                              pin_memory = True)#, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size = config.val_batch_size, shuffle = False, num_workers = config.num_workers, 
                            pin_memory = True)#, collate_fn=collate_fn)
    
    model = set_model(config, config.model.name, config.model.n_channels)
    model = model.to(config.device)
    
    scaler = GradScaler()

    if config.optimizer == 'adam':
        #TODO: Introduce config option for betas
        optimizer = Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay, betas=(config.beta1, 0.999))

    if config.optimizer == 'SGD':
        optimizer = SGD(model.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)

    #NOTE: CE loss might not be the best to use for semantic segmentation, look into jaccard losses.
    if config.loss_function =='CE':
        train_loss = nn.CrossEntropyLoss()
        eval_loss = nn.CrossEntropyLoss()   

    if config.loss_function == 'teacher_student_loss':
        train_loss = teacher_student_loss(config.model.teacher_student.teacher_weight,
                                          config.model.teacher_student.ts_loss,
                                          config.model.teacher_student.rep_layer,
                                          config.model.teacher_student.rep_weight)
        eval_loss = smp.losses.TverskyLoss(mode='multiclass')

    if config.loss_function == 'tversky':
        train_loss = smp.losses.TverskyLoss(mode='multiclass')
        eval_loss = smp.losses.TverskyLoss(mode='multiclass')
    
    if config.loss_function == 'senti_loss':
        train_loss = senti_loss()
        eval_loss = senti_loss()
        
    if config.model.name == 'teacher_student':
        teacher = get_teacher(config, config.model.teacher_student.teacher_path, config.model.teacher_student.teacher_model_type)
        teacher.to(config.device)

    print(train_loss)
    print(eval_loss)

    epoch = 0
    best_val_loss = np.inf
    CE_loss = nn.CrossEntropyLoss() 

    while epoch < config.max_epochs:
        print(torch.cuda.current_device())
        print(torch.cuda.is_available())
        print(next(model.parameters()).device)
        
        print('Epoch: '+str(epoch))
        epoch_loss = [] 
        model.train()
        if config.model.name == 'teacher_student':
            teacher.eval()

        train_iter = iter(train_loader)
        y_pred_list = [] #list to save for an entire epoch
        y_list = []
        
        for batch in tqdm(train_iter):
            if config.dataset.using_senti:
                x, y, senti = batch
                senti = senti.to(config.device)
            else:
                x, y = batch

            x = x.to(config.device) # dtype=torch.float32)
            y = y.to(config.device)   
                      
            with autocast():
                if config.model.name == 'resnet50':
                    y_pred = model(x)['out'] #NOTE: dlv3_r50 returns a dictionary
                elif config.model.name == 'unet':
                    print(model.encoder(x)[-1].shape)
                    y_pred = model(x)
                elif config.model.name =='unet_senti' or config.model.name == 'unet_senti_utae':
                    y_pred, y_pred_senti = model(x, senti)
                elif config.model.name == 'teacher_student':
                    model,y_pred,l = teacher_student(teacher, model, 'train', train_loss, x, y, config.model.teacher_student.teacher_channels, config.model.teacher_student.rep_layer)
                  
                if config.loss_function == 'senti_loss':
                    l = train_loss(y_pred, y_pred_senti, y)
                elif config.model.name != 'teacher_student':
                    l = train_loss(y_pred, y)
            
            y_pred = torch.argmax(y_pred, dim=1)
            optimizer.zero_grad()
            scaler.scale(l).backward()
            scaler.step(optimizer)
            scaler.update()
            
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
            if config.model.name == 'teacher_student':
                teacher.eval()

            val_loss = []
            CE_val_loss =[]
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

                if config.dataset.using_senti:
                    x, y, senti = batch
                    senti = senti.to(config.device)
                else:
                    x, y = batch

                x = x.to(config.device) # dtype=torch.float32)
                y = y.to(config.device)    
                with torch.no_grad():
                    if config.model.name == 'resnet50':
                        y_pred = model(x)['out'] #NOTE: dlv3_r50 returns a dictionary
                    elif config.model.name == 'unet':
                        y_pred = model(x)     
                    elif config.model.name =='unet_senti' or config.model.name == 'unet_senti_utae':
                        y_pred, y_pred_senti = model(x, senti)
                    elif config.model.name == 'teacher_student':
                        model,y_pred,l = teacher_student(teacher, model, 'val', eval_loss, x, y, config.model.teacher_student.teacher_channels, 'False')
                        

                    if config.loss_function == 'senti_loss':
                        l = eval_loss(y_pred, y_pred_senti, y)
                    elif config.model.name != 'teacher_student':
                        l = eval_loss(y_pred, y)
                    
                l_CE =  CE_loss(y_pred, y)
                y_pred = torch.argmax(y_pred, dim=1)
                val_loss.append(l.item())
                CE_val_loss.append(l_CE.item()) 
        
                if counter in idx_list and epoch % 30 == 0:
                    x_cpu =  x[0, :, :, :].cpu().detach().contiguous().numpy()
                    #senti_cpu = senti[0, 6, :, :, :].cpu().detach().contiguous().numpy() # only july
                    y_pred_cpu = y_pred[0, :, :].to(torch.uint8).cpu().detach().contiguous().numpy()
                    y_cpu = y[0, :, :].to(torch.uint8).cpu().detach().contiguous().numpy()
                    save_image(counter, x_cpu, y_pred_cpu, y_cpu, epoch, config, writer)
                    #save_senti_image(counter, senti_cpu, epoch, config, writer)                    

                y_pred = y_pred.to(torch.uint8).cpu().contiguous().detach().numpy()
                y = y.to(torch.uint8).cpu().contiguous().detach().numpy()
                val_y_pred_list.append(y_pred)
                val_y_list.append(y)            

                counter += 1
               
            #Save loss
            l_val = np.mean(val_loss)
            writer.add_scalar('val/loss', l_val, epoch)
            print('Val loss: '+str(l_val))

            CE_val = np.mean(CE_val_loss)
            writer.add_scalar('val/Cross_Entropy', CE_val, epoch)
            print('Val CE: '+str(CE_val))

            miou_prec_rec_writing(config, val_y_pred_list, val_y_list, part='val', writer=writer, epoch=epoch)
            miou_prec_rec_writing_13(config, val_y_pred_list, val_y_list, part='val', writer=writer, epoch=epoch)

            #if epoch % 15 == 0:
            #    conf_matrix(config, val_y_pred_list, val_y_list, writer, epoch)

            del val_y_list, val_y_pred_list, y_pred, y

            torch.cuda.empty_cache()

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