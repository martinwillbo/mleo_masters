import torch
import torch.nn as nn
from support_functions import set_model
from support_functions_logging import miou_prec_rec_writing, miou_prec_rec_writing_13, conf_matrix, save_image
from torch.utils.data import DataLoader
import util
from torchvision.models.segmentation.deeplabv3 import deeplabv3_resnet50
import os
from tqdm import tqdm
import numpy as np
from support_functions_noise import zero_out, set_noise, pixel_wise_fade
import segmentation_models_pytorch as smp

def eval_model(config, writer, training_path, eval_type):
    dataset_module = util.load_module(config.dataset.script_location)

    val_set = dataset_module.val_set(config)
    val_loader = DataLoader(val_set, batch_size = config.val_batch_size, shuffle = False, num_workers = config.num_workers,
                            pin_memory = True)

    #model=torch.load()
    #model = set_model(config, config.model.name, config.model.n_channels)
    #model.to(config.device)

    #Load and overwrite model
    saved_model_path = os.path.join(training_path, 'best_model.pth')
    print(saved_model_path)
    #model.load_state_dict(torch.load(saved_model_path), 'cuda:0')
    model.load(saved_model_path, 'cuda:0')
    #Set weights to 0
    if eval_type == "zero_out":
        with torch.no_grad():
        
            model = zero_out(noise_level=1.0, model=model, three_five=False)
            #model.backbone.conv1.weight[:, 3:5, :, :] = 0

    if eval_type == "zero_out_5/3":
        with torch.no_grad():
            model = zero_out(noise_level=1.0, model=model, three_five=True)
            #model.backbone.conv1.weight[:, 3:5, :, :] = 0
            #model.backbone.conv1.weight[:, 0:3, :, :] *= 5/3 #size up weights 
    
    model.eval()

    if config.loss_function == "CE":
        eval_loss_f = nn.CrossEntropyLoss()
    
    if config.loss_function == 'tversky':
        eval_loss_f = smp.losses.TverskyLoss(mode='multiclass')

    if config.loss_function == 'teacher_student_loss':
        eval_loss_f = smp.losses.TverskyLoss(mode='multiclass')


    eval_loss = []
    val_iter = iter(val_loader)
    y_pred_list = []
    y_list = []
    correct_probs = []
    incorrect_probs = []
    correct_sum = 0
    incorrect_sum = 0

    idx_list = [1,10,40]
    c = 0
    noise_level = 1.0 #want it to be only noise
    for batch in tqdm(val_iter):
        x, y = batch

        #if eval_type == 'zero_out' or eval_type == 'zero_out_5/3': #make sure no data is inputed
        #    x[:,3:5,:,:] = 0

        #if eval_type != 'normal':
        #    x = set_noise(config, x, noise_level, eval_type)

        x = x.to(config.device)
        y = y.to(config.device)

        with torch.no_grad():
            #y_pred = model(x)['out']
            y_pred = model(x)
            

        l = eval_loss_f(y_pred, y)
        eval_loss.append(l.item())

        # Extract probabilities for the predicted class
        y_prob = torch.softmax(y_pred, dim=1)
        predicted_class_probs = torch.max(y_prob, dim=1)[0]
        #print(predicted_class_probs.shape)

        #Prediction
        y_pred = torch.argmax(y_pred, dim=1)
    
        # Identify correct and incorrect predictions
        correct_mask = (y_pred == y)
        incorrect_mask = ~correct_mask
        correct_sum += torch.sum(correct_mask)
        incorrect_sum += torch.sum(incorrect_mask)

        # Calculate average probability for correct predictions
        if torch.sum(correct_mask) > 0:
            correct_avg_prob = torch.mean(predicted_class_probs[correct_mask])
            correct_avg_prob.to(torch.uint8).cpu().contiguous().numpy()
            correct_probs.append(correct_avg_prob.item())
        
        # Calculate average probability for incorrect predictions
        if torch.sum(incorrect_mask) > 0:
            incorrect_avg_prob = torch.mean(predicted_class_probs[incorrect_mask])
            incorrect_avg_prob.to(torch.uint8).cpu().contiguous().numpy()
            incorrect_probs.append(incorrect_avg_prob.item())

        #if c in idx_list:
        #    x_cpu =  x[0, :, :, :].cpu().detach().contiguous().numpy()
        #    y_pred_cpu = y_pred[0, :, :].to(torch.uint8).cpu().detach().contiguous().numpy()
        #    y_cpu = y[0, :, :].to(torch.uint8).cpu().detach().contiguous().numpy()
        #    save_image(c, x_cpu, y_pred_cpu, y_cpu, 0, config, writer)

        y_pred = y_pred.to(torch.uint8).cpu().contiguous().numpy()
        y = y.to(torch.uint8).cpu().contiguous().numpy()
        y_pred_list.append(y_pred)
        y_list.append(y)
        c += 1

    l_test = np.mean(eval_loss)
    print("loss: " + str(l_test))
    #writer.add_text("evaluation/noise level", str(noise_level), 0)
    #writer.add_scalar('evaluation/loss', l_test)
    #miou_prec_rec_writing(config, y_pred_list, y_list, 'evaluation', writer, 0)
    #miou_prec_rec_writing_13(config, y_pred_list, y_list, 'evaluation', writer, 0)

    # Calculate average probabilities
    avg_correct_prob = sum(correct_probs) / len(correct_probs)
    avg_incorrect_prob = sum(incorrect_probs) / len(incorrect_probs) 

    #Accuracy
    acc = correct_sum / (correct_sum + incorrect_sum)

    print("Average Accuracy:", acc.item())
    print("Average Probability for Correct Predictions:", avg_correct_prob)
    print("Average Probability for Incorrect Predictions:", avg_incorrect_prob)
    #conf_matrix(config, y_pred_list, y_list, writer, 0)
