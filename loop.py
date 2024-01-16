import torch
import torch.nn as nn
import numpy as np
import util
from tqdm import tqdm
from torchvision.models.segmentation.deeplabv3 import deeplabv3_resnet50
from torchvision.models.segmentation.deeplabv3 import DeepLabV3, DeepLabHead
from torch.utils.data import DataLoader
from torch.optim import Adam

def loop(config, train_set, val_set, writer = None):

    #dataset_module = util.load_module(config.dataset.script_location)
    #train_set = dataset_module.train_set(config)
    #val_set = dataset_module.val_set(config)
    #NOTE: Just outlining the 'interface' of this way of structuring the experiment code
    #test_set = dataset_module.test_set(config)

    train_loader = DataLoader(train_set, batch_size = config.batch_size, shuffle = True, num_workers = config.num_workers,
                              pin_memory = True)
    val_loader = DataLoader(val_set, batch_size = config.val_batch_size, shuffle = False, num_workers = config.num_workers, 
                            pin_memory = True)
    
    #NOTE: There is a implementation difference here between dataset and model, we could have used the same scheme for the model.
    #Just showcasing two ways of doing things. This approach is 'simpler' but offers less modularity (which is always not bad).
    #If we intend to mainly work with one model and don't need to wrap it in custom code or whatever this is fine.
    model = deeplabv3_resnet50(weights = config.model.pretrained, progress = False, num_classes = config.model.n_class,
                                dim_input = config.model.n_channels, aux_loss = None, weights_backbone = config.model.pretrained_backbone)


    if config.optimizer == 'adam':
        #TODO: Introduce config option for betas
        optimizer = Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay, betas=(config.beta1, 0.999))

    if config.use_transform:
        transform_module = util.load_module(config.transform.script_location)
        transform = transform_module.get_transform(config)
        train_set.set_transform(transform)

    #NOTE: CE loss might not be the best to use for semantic segmentation, look into jaccard losses.
    train_loss = nn.CrossEntropyLoss()
    eval_loss = nn.CrossEntropyLoss()
    
    epoch = 0
    best_val_loss = np.inf

    #NOTE: Can also include a check for early stopping counter here.
    while epoch < config.max_epochs:
        
        print('Epoch: '+str(epoch))
        epoch_loss = []
        epoch_miou_prec_rec = []
        model.train()
        train_iter = iter(train_loader)
        for batch in tqdm(train_iter):
            optimizer.zero_grad()
            x, y = batch
            x = x.to(config.device)
            y = y.to(config.device)
            #NOTE: dlv3_r50 returns a dictionary
            y_pred = model(x)['out']
            l = train_loss(y_pred, y)
            l.backward()
            optimizer.step()
            #NOTE: If you have a learning rate scheduler this is to place to step it. 
            y_pred = torch.argmax(y_pred, dim=1)
            y_pred = y_pred.cpu().contiguous()
            y = y.cpu().contiguous()
            y_pred_flat = y_pred.view(-1).numpy()
            y_flat = y.view(-1).numpy()
            iou_prec_rec = np.nan * np.empty((3, config.model.n_class))
            for i in range(config.model.n_class):
                y_flat_i = y_flat == i
                num_i = np.count_nonzero(y_flat_i)
                pred_flat_i = y_pred_flat == i
                num_pred_i = np.count_nonzero(pred_flat_i)
                intersection_i = np.logical_and(y_flat_i, pred_flat_i)
                union_i = np.logical_or(y_flat_i, pred_flat_i)
                num_intersection_i = np.count_nonzero(intersection_i)
                num_union_i = np.count_nonzero(union_i)
                if num_union_i > 0:
                    iou_prec_rec[0,i] = num_intersection_i/num_union_i
                if num_pred_i > 0:
                    iou_prec_rec[1,i] = num_intersection_i / num_pred_i
                if num_i > 0:
                    iou_prec_rec[2,i] = num_intersection_i / num_i

            epoch_miou_prec_rec.append(iou_prec_rec)
            epoch_loss.append(l.item())
        
        epoch_miou_prec_rec = np.nanmean(np.stack(epoch_miou_prec_rec, axis = 0), axis = 0)
        writer.add_scalar('train/loss', np.mean(epoch_loss), epoch)
        print('Epoch mean loss: '+str(np.mean(epoch_loss)))
        writer.add_scalar('train/miou', np.mean(epoch_miou_prec_rec[0,:]), epoch)
        writer.add_scalar('train/precision', np.mean(epoch_miou_prec_rec[1,:]), epoch)
        writer.add_scalar('train/recall', np.mean(epoch_miou_prec_rec[2,:]), epoch)
        print('Epoch mean miou: '+str(np.mean(epoch_miou_prec_rec[0,:])))
        print('Epoch mean precision: '+str(np.mean(epoch_miou_prec_rec[1,:])))
        print('Epoch mean recall: '+str(np.mean(epoch_miou_prec_rec[2,:])))

        if epoch % config.eval_every == 0:
            #NOTE: I added some basic eval code here. Typically I hide it in a module, depends on how much it is.
            model.eval()
            val_loss = []
            val_miou_prec_rec = []
            val_iter = iter(val_loader)
            for batch in tqdm(val_iter):
                x, y = batch
                x = x.to(config.device)
                y = y.to(config.device)
                y_pred = model(x)['out']
                y_pred = torch.argmax(y_pred, dim=1)
                y_pred = y_pred.cpu().contiguous()
                y = y.cpu().contiguous()
                y_pred_flat = y_pred.view(-1).numpy()
                y_flat = y.view(-1).numpy()
                iou_prec_rec = np.nan * np.empty((3, config.model.n_class))
                for i in range(config.model.n_class):
                    y_flat_i = y_flat == i
                    num_i = np.count_nonzero(y_flat_i)
                    pred_flat_i = y_pred_flat == i
                    num_pred_i = np.count_nonzero(pred_flat_i)
                    intersection_i = np.logical_and(y_flat_i, pred_flat_i)
                    union_i = np.logical_or(y_flat_i, pred_flat_i)
                    num_intersection_i = np.count_nonzero(intersection_i)
                    num_union_i = np.count_nonzero(union_i)
                    if num_union_i > 0:
                        iou_prec_rec[0,i] = num_intersection_i/num_union_i
                    if num_pred_i > 0:
                        iou_prec_rec[1,i] = num_intersection_i / num_pred_i
                    if num_i > 0:
                        iou_prec_rec[2,i] = num_intersection_i / num_i
                val_miou_prec_rec.append(iou_prec_rec)
                val_loss.append(l.item())

            val_miou_prec_rec = np.nanmean(np.stack(val_miou_prec_rec, axis = 0), axis = 0)
            l_val = np.mean(val_loss)
            writer.add_scalar('val/loss', l_val, epoch)
            writer.add_scalar('val/miou', np.mean(val_miou_prec_rec[0,:]), epoch)
            writer.add_scalar('val/precision', np.mean(val_miou_prec_rec[1,:]), epoch)
            writer.add_scalar('val/recall', np.mean(val_miou_prec_rec[2,:]), epoch)
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