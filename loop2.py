import torch
import torch.nn as nn
import numpy as np
import util
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.cuda.amp import autocast, GradScaler
import os
import math
import random
import segmentation_models_pytorch as smp
from support_functions_logging import miou_prec_rec_writing, miou_prec_rec_writing_13, conf_matrix, save_image, save_senti_image, label_image
from support_functions_noise import set_noise, zero_out, stepwise_linear_function_1, stepwise_linear_function_2, custom_sine, image_wise_fade
from support_functions_loop import set_model, set_loss, get_loss_y_pred, get_teacher, teacher_student, multi_teacher



def loop2(config, writer, hydra_log_dir):
    dataset_module = util.load_module(config.dataset.script_location)
    train_set = dataset_module.train_set(config)
    val_set = dataset_module.val_set(config)
    
    #save label image for reference
    label_image(config, writer)
    
    train_loader = DataLoader(train_set, batch_size = config.batch_size, shuffle = True, num_workers = config.num_workers,
                              pin_memory = True)
    val_loader = DataLoader(val_set, batch_size = config.val_batch_size, shuffle = False, num_workers = config.num_workers, 
                            pin_memory = True)
    
    
    if config.model.name == 'teacher_student':
        teacher = get_teacher(config, config.model.teacher_student.teacher_path, config.model.teacher_student.teacher_channels)
        teacher.to(config.device)
        model = set_model(config, config.model.teacher_student.student_name, config.model.teacher_student.student_channels)
    elif config.model.name == 'multi_teacher':
        teacher_1 = get_teacher(config, config.model.multi_teacher.teacher_1_path, config.model.multi_teacher.teacher_1_channels)
        teacher_2 = get_teacher(config, config.model.multi_teacher.teacher_2_path, config.model.multi_teacher.teacher_2_channels)
        teacher_1.to(config.device)
        teacher_2.to(config.device)
        model = set_model(config, config.model.multi_teacher.student_name, config.model.multi_teacher.student_channels)
    elif config.model.name == 'unet_predict_priv':
        model = set_model(config, config.model.name, config.model.unet_predict_priv.unet_channels)
    else:
        model = set_model(config, config.model.name, config.model.n_channels)

    model.to(config.device)
    scaler = GradScaler()
    

    if config.optimizer == 'adam':
        #TODO: Introduce config option for betas
        optimizer = Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay, betas=(config.beta1, 0.999))

    if config.optimizer == 'SGD':
        optimizer = SGD(model.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)

    #NOTE: CE loss might not be the best to use for semantic segmentation, look into jaccard losses.
        
    train_loss, eval_loss = set_loss(config.loss_function, config)
    CE_loss, tversky_loss = nn.CrossEntropyLoss(), smp.losses.TverskyLoss(mode='multiclass')

    epoch = 0
    best_val_loss = np.inf

    while epoch < config.max_epochs:
        print(torch.cuda.current_device())
        print(torch.cuda.is_available())
        print(next(model.parameters()).device)
        
        print('Epoch: '+str(epoch))
        epoch_loss = [] 
        model.train()
        if config.model.name == teacher_student:
            teacher.eval()
        
        train_iter = iter(train_loader)

        y_pred_list = [] #list to save for an entire epoch
        y_list = []
        if config.noise.stepwise_linear_function == 'function_1':
            noise_level = stepwise_linear_function_1(epoch, config.max_epochs)
        elif config.noise.stepwise_linear_function == 'function_2':
            noise_level = stepwise_linear_function_2(epoch, config.max_epochs)
        elif config.noise.stepwise_linear_function == 'sine':
            noise_level = custom_sine(epoch)
            print(noise_level)

        
        for batch in tqdm(train_iter):
            x,y, mtd,senti = batch

            if config.noise.noise:

                if config.noise.noise_type == 'zero_out':
                    model= zero_out(noise_level, model)
            
                elif config.noise.noise_distribution_type == 'image':
                    x = set_noise(x, noise_level, config.noise.noise_type)

                elif config.noise.noise_distribution_type == 'batch':
                    num_rows_to_noise = math.ceil(noise_level * x.shape[0])
                    rows_to_noise = random.sample(range(x.shape[0]), num_rows_to_noise)
                    x[rows_to_noise, :, :, :] = set_noise(x[rows_to_noise, :, :, :], noise_level, config.noise.noise_type)

                else:
                    print("Invalid noise_distribution_type or noise_type")

            x = x.to(config.device)
            y = y.to(config.device)
            mtd = mtd.to(config.device)
            senti = senti.to(config.device)
            
            with autocast():
                if config.model.name == 'teacher_student':
                    model, y_pred, l = teacher_student(teacher, model, 'train', train_loss, x, y, 
                                                       config.model.teacher_student.student_spec_channels,
                                                       config.model.teacher_student.teacher_spec_channels)
                
                elif config.model.name == 'multi_teacher':
                    model, y_pred, l = multi_teacher(teacher_1, teacher_2, model, 'train', train_loss, x, y, 
                                                     config.model.multi_teacher.student_spec_channels,
                                                     config.model.multi_teacher.teacher_1_spec_channels,
                                                     config.model.multi_teacher.teacher_2_spec_channels)
                elif config.model.name == 'unet_predict_priv':
                    y_pred, x_priv_pred = model(x)
                    l = train_loss(y_pred, x_priv_pred, y, x[:,3:,:,:])
                else:
                    model, y_pred, l = get_loss_y_pred(config.model.name, config.loss_function, train_loss, model, x, mtd, senti, y)
            optimizer.zero_grad()
            scaler.scale(l).backward()
            scaler.step(optimizer)
            scaler.update()

            y_pred = torch.argmax(y_pred, dim=1) #sets class to each data point

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

            if config.model.name == teacher_student:
                teacher.eval()
            
            val_loss = []
            val_CE_loss, val_tversky_loss = [], []
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
                x,y, mtd, senti = batch

                if config.noise.noise:
                    if config.noise.noise_type == 'zero_out':
                        #test:)))
                        x[:, 3:5, :, :] = image_wise_fade(x[:, 3:5, :, :], 1.0)
                        model = zero_out(1.0, model)
                    else:
                        x = set_noise(x, 1.0, config.noise.noise_type)

                x = x.to(config.device)
                y = y.to(config.device)
                mtd = mtd.to(config.device)
                senti = senti.to(config.device)

                with torch.no_grad():
                    if config.model.name == 'teacher_student':
                        model, y_pred, l = teacher_student(teacher, model, 'val', eval_loss, x, y, 
                                                        config.model.teacher_student.student_spec_channels,
                                                        config.model.teacher_student.teacher_spec_channels)
                    elif config.model.name == 'multi_teacher':
                        model, y_pred, l = multi_teacher(teacher_1, teacher_2, model, 'val', eval_loss, x, y, 
                                                     config.model.multi_teacher.student_spec_channels,
                                                     config.model.multi_teacher.teacher_1_spec_channels,
                                                     config.model.multi_teacher.teacher_2_spec_channels)
                    elif config.model.name == 'unet_predict_priv':
                        y_pred, x_priv_pred = model(x)
                        l = eval_loss(y_pred, y)
                    else:
                        model, y_pred, l = get_loss_y_pred(config.model.name, config.loss_function, eval_loss, model, x, mtd, senti, y)

                CE_l, tversky_l = CE_loss(y_pred, y), tversky_loss(y_pred, y)
                y_pred = torch.argmax(y_pred, dim=1)
                val_loss.append(l.item())
                val_CE_loss.append(CE_l.item())
                val_tversky_loss.append(tversky_l.item())
        
                if counter in idx_list and epoch % 15 == 0:
                    x_cpu =  x[0, :, :, :].cpu().detach().contiguous().numpy()
                    y_pred_cpu = y_pred[0, :, :].to(torch.uint8).cpu().detach().contiguous().numpy()
                    y_cpu = y[0, :, :].to(torch.uint8).cpu().detach().contiguous().numpy()
                    senti_cpu = senti[0, 6, :, :, :].cpu().detach().contiguous().numpy()
                    save_image(counter, x_cpu, y_pred_cpu, y_cpu, epoch, config, writer)
                    save_senti_image(counter, senti_cpu, epoch, config, writer)

                y_pred = y_pred.to(torch.uint8).cpu().contiguous().detach().numpy()
                y = y.to(torch.uint8).cpu().contiguous().detach().numpy()
                val_y_pred_list.append(y_pred)
                val_y_list.append(y)

                counter += 1

            #Save loss
            l_val = np.mean(val_loss)
            l_CE_val = np.mean(val_CE_loss)
            l_tversky_val = np.mean(val_tversky_loss)
            writer.add_scalar('val/loss', l_val, epoch)
            writer.add_scalar('loss/CE', l_CE_val, epoch)
            writer.add_scalar('loss/tversky', l_tversky_val, epoch)
            print('Val loss: '+str(l_val))

            miou_prec_rec_writing(config, val_y_pred_list, val_y_list, part='val', writer=writer, epoch=epoch)
            miou_prec_rec_writing_13(config, val_y_pred_list, val_y_list, part='val', writer=writer, epoch=epoch)

            if epoch % 15 == 0:
                conf_matrix(config, val_y_pred_list, val_y_list, writer, epoch)

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

        


    




    


        