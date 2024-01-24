import torch
import torch.nn as nn
import numpy as np
import util
from tqdm import tqdm
from torch.utils.data import DataLoader
from loop2 import miou_prec_rec_writing, miou_prec_rec_writing_13 
import os

def eval_on_test(config, writer, training_path):

    dataset_module = util.load_module(config.dataset.script_location)
    
    #NOTE: Just outlining the 'interface' of this way of structuring the experiment code
    test_set = dataset_module.test_set(config)

    test_loader = DataLoader(test_set, batch_size = config.test_batch_size, shuffle = False, num_workers = config.num_workers, 
                            pin_memory = True)
    
    # Specify the path to the saved model
    saved_model_path = os.path.join(training_path, 'best_model.pth')

    # Load the saved model parameters into the instantiated model
    model = model.load_state_dict(torch.load(saved_model_path))
    model.to(config.device)

    # Set the model to evaluation mode
    model.eval()

    test_loss_f = nn.CrossEntropyLoss()

    test_loss = []
    test_miou_prec_rec = []
    test_iter = iter(test_loader)
    y_pred_list = [] #list to save for an entire epoch
    y_list = []

    for batch in tqdm(test_iter):
        x, y = batch
        x = x.to(config.device)
        y = y.to(config.device)
        
        with torch.no_grad():
            #y_pred = model(x)['out']
            y_pred = model(x)

        l = test_loss_f(y_pred, y)    
        test_loss.append(l.item()) 

        y_pred = torch.argmax(y_pred, dim=1)
        y_pred = y_pred.to(torch.uint8).cpu().contiguous().numpy()
        y = y.to(torch.uint8).cpu().contiguous().numpy()
        y_pred_list.append(y_pred)
        y_list.append(y)
                 
    l_test = np.mean(test_loss)

    # Assuming 'writer' is defined somewhere in your code for logging
    writer.add_scalar('test/loss', l_test)
    miou_prec_rec_writing(config, y_pred_list, y_list, 'test', writer, 0)
    miou_prec_rec_writing_13(y_pred_list, y_list, 'test', writer, 0) 