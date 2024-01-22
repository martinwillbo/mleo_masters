import torch
import torch.nn as nn
import numpy as np
import util
from tqdm import tqdm
from torchvision.models.segmentation.deeplabv3 import deeplabv3_resnet50
#from torchvision.models.segmentation.deeplabv3 import DeepLabV3, DeepLabHead
from torch.utils.data import DataLoader
def eval_on_test(config, writer = None):
    dataset_module = util.load_module(config.dataset.script_location)
    #NOTE: Just outlining the 'interface' of this way of structuring the experiment code
    test_set = dataset_module.test_set(config)
    test_loader = DataLoader(test_set, batch_size = config.test_batch_size, shuffle = False, num_workers = config.num_workers,
                            pin_memory = True)
    model = deeplabv3_resnet50(weights = config.model.pretrained, progress=True, num_classes=config.model.n_class,
                                  dim_input=config.model.n_channels, aux_loss=None, weights_backbone=config.model.pretrained_backbone)
    model.to(config.device)
    # Specify the path to the saved model
    saved_model_path = 'path/to/your/saved_model.pth'
    # Load the saved model parameters into the instantiated model
    model = model.load_state_dict(torch.load(saved_model_path))
    # Set the model to evaluation mode
    model.eval()
    test_loss_f = nn.CrossEntropyLoss()
    test_loss = []
    test_miou_prec_rec = []
    test_iter = iter(test_loader)
    for batch in tqdm(test_iter):
        x, y = batch
        x = x.to(config.device)
        y = y.to(config.device)
        with torch.no_grad():
            y_pred = model(x)['out']
        l = test_loss_f(y_pred, y)
        test_loss.append(l.item())
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
                iou_prec_rec[0, i] = num_intersection_i / num_union_i
            if num_pred_i > 0:
                iou_prec_rec[1, i] = num_intersection_i / num_pred_i
            if num_i > 0:
                iou_prec_rec[2, i] = num_intersection_i / num_i
        test_miou_prec_rec.append(iou_prec_rec)
    test_miou_prec_rec = np.nanmean(np.stack(test_miou_prec_rec, axis=0), axis=0)
    l_test = np.mean(test_loss)
    # Assuming 'writer' is defined somewhere in your code for logging
    writer.add_scalar('test/loss', l_test)
    writer.add_scalar('test/miou', np.mean(test_miou_prec_rec[0, :]))
    writer.add_scalar('test/precision', np.mean(test_miou_prec_rec[1, :]))
    writer.add_scalar('test/recall', np.mean(test_miou_prec_rec[2, :]))
    print('Test loss: ' + str(l_test))
    print('Test mean miou: ' + str(np.mean(test_miou_prec_rec[0, :])))
    print('Test mean precision: ' + str(np.mean(test_miou_prec_rec[1, :])))
    print('Test mean recall: ' + str(np.mean(test_miou_prec_rec[2, :])))