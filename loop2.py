import torch
import torch.nn as nn
import numpy as np
import util
from tqdm import tqdm
from torchvision.models.segmentation.deeplabv3 import deeplabv3_resnet50
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.cuda.amp import autocast, GradScaler
from fcnpytorch.fcn8s import FCN8s as FCN8s #smaller net!
import os
import math
import random
import segmentation_models_pytorch as smp
from support_functions_logging import miou_prec_rec_writing, miou_prec_rec_writing_13, conf_matrix, save_image, label_image
from support_functions_noise import set_noise, zero_out, stepwise_linear_function_1, stepwise_linear_function_2, custom_sine, image_wise_fade
from unet_module import UNetWithMetadata, UnetFeatureMetadata, UnetFeatureMetadata_2
from CE_tversky_loss import CE_tversky_Loss

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
    
    if config.model.name == 'resnet50':
    
        model = deeplabv3_resnet50(weights = config.model.pretrained, progress = True, #num_classes = config.model.n_class,
                                    dim_input = config.model.n_channels, aux_loss = None, weights_backbone = config.model.pretrained_backbone)
        
        model.classifier[4] = torch.nn.Conv2d(256, config.model.n_class, kernel_size=(1,1), stride=(1,1))
        if config.dropout:
            dropout_rate = 0.5  # Example dropout rate, adjust as needed
            classifier = list(model.classifier.children())
            classifier.insert(-1, nn.Dropout(dropout_rate))  # Insert dropout before the last layer in the classifier
            model.classifier = nn.Sequential(*classifier)
        
        model.backbone.conv1 = nn.Conv2d(config.model.n_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    
    elif config.model.name == 'FCN8':
        model = FCN8s(n_class=config.model.n_class, dim_input=config.model.n_channels, weight_init='normal')

    elif config.model.name == 'unet':
        model = smp.Unet(
            encoder_weights="imagenet",
            encoder_name="efficientnet-b4",
            in_channels = config.model.n_channels,
            classes= config.model.n_class
        )
    elif config.model.name == 'unet_mtd':
        model = UNetWithMetadata(n_channels=config.model.n_channels, n_class=config.model.n_class, n_metadata=6, device=config.device, reweight=config.model.reweight_late, mtd_weighting = config.model.mtd_weighting)
    elif config.model.name == 'unet_mtd_feature':
        model = UnetFeatureMetadata(n_channels=config.model.n_channels, n_class=config.model.n_class, n_metadata=6)
    elif config.model.name == 'unet_mtd_feature_2':
        model = UnetFeatureMetadata_2(n_channels=config.model.n_channels, n_class=config.model.n_class, feature_block=config.model.feature_block, linear_mtd_preprocess=config.model.linear_mtd_preprocess)

    model.to(config.device)
    scaler = GradScaler()
    

    if config.optimizer == 'adam':
        #TODO: Introduce config option for betas
        optimizer = Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay, betas=(config.beta1, 0.999))

    if config.optimizer == 'SGD':
        optimizer = SGD(model.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)

    #NOTE: CE loss might not be the best to use for semantic segmentation, look into jaccard losses.
        
    if config.loss_function == 'CE':
        train_loss = nn.CrossEntropyLoss()
        eval_loss = nn.CrossEntropyLoss()
    elif config.loss_function == 'tversky':
        train_loss = smp.losses.TverskyLoss(mode='multiclass', alpha=config.model.tversky_a, beta=config.model.tversky_b)
        eval_loss = smp.losses.TverskyLoss(mode='multiclass', alpha=config.model.tversky_a, beta=config.model.tversky_b)
        CE_loss, tversky_loss = nn.CrossEntropyLoss(), smp.losses.TverskyLoss(mode='multiclass')
    elif config.loss_function == 'CE_tversky':
        train_loss = CE_tversky_Loss()
        eval_loss = CE_tversky_Loss()
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
        train_iter = iter(train_loader)

        y_pred_list = [] #list to save for an entire epoch
        y_list = []
        if config.stepwise_linear_function == 'function_1':
            noise_level = stepwise_linear_function_1(epoch, config.max_epochs)
        elif config.stepwise_linear_function == 'function_2':
            noise_level = stepwise_linear_function_2(epoch, config.max_epochs)
        elif config.stepwise_linear_function == 'sine':
            noise_level = custom_sine(epoch)
            print(noise_level)

        
        
        for batch in tqdm(train_iter):
            x,y, mtd = batch

            if config.noise:

                if config.noise_type == 'zero_out':
                    model= zero_out(noise_level, model)
            
                elif config.noise_distribution_type == 'image':
                    x = set_noise(config, x, noise_level, config.noise_type)

                elif config.noise_distribution_type == 'batch':
                    num_rows_to_noise = math.ceil(noise_level * x.shape[0])
                    rows_to_noise = random.sample(range(x.shape[0]), num_rows_to_noise)
                    x[rows_to_noise, :, :, :] = set_noise(config, x[rows_to_noise, :, :, :], noise_level, config.noise_type)

                else:
                    print("Invalid noise_distribution_type or noise_type")

            x = x.to(config.device)
            y = y.to(config.device)
            mtd = mtd.to(config.device)
            
            with autocast():
                if config.model.name == 'resnet50':
                    y_pred = model(x)['out'] #NOTE: dlv3_r50 returns a dictionary
                elif config.model.name == 'FCN8' or config.model.name == 'unet':
                    y_pred = model(x)
                elif config.model.name == 'unet_mtd' or config.model.name == 'unet_mtd_feature' or config.model.name == 'unet_mtd_feature_2':
                    if config.model.name == 'unet_mtd':
                        y_pred = model(x, mtd, epoch)
                    else:
                        y_pred = model(x, mtd)
                l = train_loss(y_pred, y)
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
                x,y, mtd = batch

                if config.noise:
                    if config.noise_type == 'zero_out':
                        #test:)))
                        x[:, 3:5, :, :] = image_wise_fade(x[:, 3:5, :, :], 1.0)
                        model = zero_out(1.0, model)
                    else:
                        x = set_noise(config, x, 1.0, config.noise_type)

                x = x.to(config.device)
                y = y.to(config.device)
                mtd = mtd.to(config.device)

                if config.model.name == 'resnet50':
                    y_pred = model(x)['out']
                elif config.model.name == 'FCN8' or config.model.name == 'unet':
                    y_pred = model(x)
                elif config.model.name == 'unet_mtd' or config.model.name == 'unet_mtd_feature' or config.model.name == 'unet_mtd_feature_2':
                    if config.model.name == 'unet_mtd':
                        y_pred = model(x, mtd, epoch)
                    else:
                        y_pred = model(x, mtd)

                l = eval_loss(y_pred, y)
                CE_l, tversky_l = CE_loss(y_pred, y), tversky_loss(y_pred, y)
                y_pred = torch.argmax(y_pred, dim=1)
                val_loss.append(l.item())
                val_CE_loss.append(CE_l.item())
                val_tversky_loss.append(tversky_l.item())
        
                if counter in idx_list and epoch % 15 == 0:
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

        


   

    




    


        