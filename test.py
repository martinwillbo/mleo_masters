import torch
import torch.nn as nn
import numpy as np
import util
from tqdm import tqdm
from torch.utils.data import DataLoader
from support_functions_logging import miou_prec_rec_writing, miou_prec_rec_writing_13, conf_matrix
import os
from torchvision.models.segmentation.deeplabv3 import deeplabv3_resnet50
import segmentation_models_pytorch as smp


def eval_on_test(config, writer, training_path):
    dataset_module = util.load_module(config.dataset.script_location)
    #NOTE: Just outlining the 'interface' of this way of structuring the experiment code

    test_set = dataset_module.test_set(config)
    test_loader = DataLoader(test_set, batch_size = config.test_batch_size, shuffle = False, num_workers = config.num_workers,
                            pin_memory = True)
    
    # Specify the path to the saved model
    saved_model_path = os.path.join(training_path, 'best_model.pth')
    #model = deeplabv3_resnet50(weights = config.model.pretrained, progress = True, #num_classes = config.model.n_class,
                                #dim_input = config.model.n_channels, aux_loss = None, weights_backbone = config.model.pretrained_backbone)
    
    #model.classifier[4] = torch.nn.Conv2d(256, config.model.n_class, kernel_size=(1,1), stride=(1,1))
    #model.backbone.conv1 = nn.Conv2d(config.model.n_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    model = smp.Unet(
            encoder_weights="imagenet",
            encoder_name="efficientnet-b4",
            in_channels = config.model.n_channels,
            classes= config.model.n_class
        )

    #Load and overwrite model
    saved_model_path = os.path.join(training_path, 'best_model.pth')
    model.load_state_dict(torch.load(saved_model_path))
    model.to(config.device)

    # Set the model to evaluation mode
    model.eval()
    test_loss_f = smp.losses.TverskyLoss(mode='multiclass', alpha=config.model.tversky_a, beta=config.model.tversky_b)
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
    print("loss: " + str(l_test))
    # Assuming 'writer' is defined somewhere in your code for logging
    writer.add_scalar('test/loss', l_test)
    miou_prec_rec_writing(config, y_pred_list, y_list, 'test', writer, 0)
    miou_prec_rec_writing_13(config, y_pred_list, y_list, 'test', writer, 0)
    #conf_matrix(config, y_pred_list, y_list, writer, 0)






