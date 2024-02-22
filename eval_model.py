import torch
import torch.nn as nn
from loop2 import miou_prec_rec_writing, miou_prec_rec_writing_13, conf_matrix, save_image
from torch.utils.data import DataLoader
import util
from torchvision.models.segmentation.deeplabv3 import deeplabv3_resnet50
import os
from tqdm import tqdm
import numpy as np
from support_functions_noise import set_noise, zero_out
import segmentation_models_pytorch as smp
from support_functions_loop import set_model

def eval_model(config, writer, training_path, eval_type):
    dataset_module = util.load_module(config.dataset.script_location)

    val_set = dataset_module.val_set(config)
    val_loader = DataLoader(val_set, batch_size = config.val_batch_size, shuffle = False, num_workers = config.num_workers,
                            pin_memory = True)


    model = set_model(config, model_name=config.eval.eval_model, n_channels=config.eval.eval_channels)

    #Load and overwrite model
    saved_model_path = os.path.join(training_path, 'best_model.pth')
    model.load_state_dict(torch.load(saved_model_path))

    noise_level = 1.0 #want it to be only noise

    #Set weights to 0
    if eval_type == "zero_out":
        with torch.no_grad():
            model = zero_out(noise_level, model)

    if eval_type == "zero_out_5/3":
        with torch.no_grad():
            model = zero_out(noise_level, model, three_five=True)


    model.to(config.device)

    model.eval()

    if config.eval.eval_loss == "CE":
        eval_loss_f = nn.CrossEntropyLoss()
    
    if config.leval.eval_loss == 'tversky':
        eval_loss_f = smp.losses.TverskyLoss(mode='multiclass')

    eval_loss = []
    val_iter = iter(val_loader)
    y_pred_list = []
    y_list = []

    idx_list = [1,10,40]
    c = 0
    
    for batch in tqdm(val_iter):
        x, y, _ = batch

        if eval_type != 'normal':
            x = set_noise(x, noise_level, eval_type)

        x = x.to(config.device)
        y = y.to(config.device)

        with torch.no_grad():
            #y_pred = model(x)['out']
            y_pred = model(x)

        l = eval_loss_f(y_pred, y)
        eval_loss.append(l.item())

        y_pred = torch.argmax(y_pred, dim=1)

        if c in idx_list:
            x_cpu =  x[0, :, :, :].cpu().detach().contiguous().numpy()
            y_pred_cpu = y_pred[0, :, :].to(torch.uint8).cpu().detach().contiguous().numpy()
            y_cpu = y[0, :, :].to(torch.uint8).cpu().detach().contiguous().numpy()
            save_image(c, x_cpu, y_pred_cpu, y_cpu, 0, config, writer)

        y_pred = y_pred.to(torch.uint8).cpu().contiguous().numpy()
        y = y.to(torch.uint8).cpu().contiguous().numpy()
        y_pred_list.append(y_pred)
        y_list.append(y)
        c += 1

    l_test = np.mean(eval_loss)
    print("loss: " + str(l_test))
    writer.add_scalar('evaluation/loss', l_test)
    miou_prec_rec_writing(config, y_pred_list, y_list, 'evaluation', writer, 0)
    miou_prec_rec_writing_13(config, y_pred_list, y_list, 'evaluation', writer, 0)
    #conf_matrix(config, y_pred_list, y_list, writer, 0)
