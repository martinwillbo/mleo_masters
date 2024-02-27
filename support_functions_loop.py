from unet_module import UNetWithMetadata, UnetFeatureMetadata, UnetFeatureMetadata_2, UnetFeatureSenti, UnetSentiDoubleLoss, UnetFeatureSentiMtd
import segmentation_models_pytorch as smp
from fcnpytorch.fcn8s import FCN8s as FCN8s #smaller net!
from torchvision.models.segmentation.deeplabv3 import deeplabv3_resnet50
import torch
import torch.nn as nn
import os

from custom_losses import CE_tversky_Loss, senti_loss, teacher_student_loss


def set_model(config, model_name, n_channels):
    if model_name == 'resnet50':
        model = deeplabv3_resnet50(weights = config.model.pretrained, progress = True, #num_classes = config.model.n_class,
                                    dim_input = n_channels, aux_loss = None, weights_backbone = config.model.pretrained_backbone)  
        model.classifier[4] = nn.Conv2d(256, config.model.n_class, kernel_size=(1,1), stride=(1,1))    
        model.backbone.conv1 = nn.Conv2d(n_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    
    elif model_name == 'FCN8':
        model = FCN8s(n_class=config.model.n_class, dim_input=n_channels, weight_init='normal')

    elif model_name== 'unet':
        model = smp.Unet(
            encoder_weights="imagenet",
            encoder_name="efficientnet-b4",
            in_channels = config.model.n_channels,
            classes= config.model.n_class
        )

    elif model_name== 'unet_mtd':
        model = UNetWithMetadata(n_channels=n_channels, n_class=config.model.n_class, n_metadata=6, device=config.device, reweight=config.model.reweight_late, mtd_weighting = config.model.mtd_weighting)
    
    elif model_name == 'unet_mtd_feature':
        model = UnetFeatureMetadata(n_channels=n_channels, n_class=config.model.n_class, n_metadata=6)
    
    elif model_name == 'unet_mtd_feature_2':
        model = UnetFeatureMetadata_2(n_channels=n_channels, n_class=config.model.n_class, feature_block=config.model.feature_block, linear_mtd_preprocess=config.model.linear_mtd_preprocess)
    
    elif config.model.name == 'unet_senti':
        model = UnetFeatureSenti(n_channels=n_channels, n_senti_channels=120, n_classes=config.model.n_class)

    elif config.model.name == 'unet_senti_double':
        model = UnetSentiDoubleLoss(n_channels=n_channels, n_senti_channels=120, n_classes=config.model.n_class)

    elif config.model.name == 'unet_senti_mtd':
        model = UnetFeatureSentiMtd(n_channels=n_channels, n_senti_channels=120, n_metadata=6, n_classes=config.model.n_class, w=config.model.mtd_weighting)

    return model

def set_loss(loss_function):
    if loss_function == 'CE':
        train_loss = nn.CrossEntropyLoss()
        eval_loss = nn.CrossEntropyLoss()
    elif loss_function == 'tversky':
        train_loss = smp.losses.TverskyLoss(mode='multiclass')
        eval_loss = smp.losses.TverskyLoss(mode='multiclass')
    elif loss_function == 'CE_tversky':
        train_loss = CE_tversky_Loss()
        eval_loss = CE_tversky_Loss()
    elif loss_function == 'senti_loss':
        train_loss = senti_loss()
        eval_loss = senti_loss()
    elif loss_function == 'teacher_student_loss':
        train_loss = teacher_student_loss()
        eval_loss = smp.losses.TverskyLoss(mode='multiclass') #let eval loss be for only student
        
    return train_loss, eval_loss

def get_loss_y_pred(model_name, loss_function, loss, model, x, mtd, senti, y):
    if model_name == 'resnet50':
        y_pred = model(x)['out'] #NOTE: dlv3_r50 returns a dictionary
    elif model_name == 'FCN8' or model_name == 'unet':
        y_pred = model(x)
    elif model_name == 'unet_mtd' or model_name == 'unet_mtd_feature' or model_name == 'unet_mtd_feature_2':
        y_pred = model(x, mtd)
    elif model_name== 'unet_senti':
        y_pred = model(x,senti)
    elif model_name == 'unet_senti_mtd':
        y_pred = model(x, senti, mtd)
    elif model_name == 'unet_senti_double':
        y_pred, y_pred_senti = model(x, senti)

    if loss_function == 'senti_loss':
        l = loss(y_pred, y_pred_senti, y)
    else:
        l = loss(y_pred, y)

    return model, y_pred, l

def get_teacher(config, teacher_path, teacher_model_type='unet') :
    teacher = set_model(config, teacher_model_type, config.n_channles)
    teacher_path = os.path.join(teacher_path, 'best_model.pth')
    teacher.load_state_dict(torch.load(teacher_path))
    return teacher


def teacher_student(teacher, student, part, loss, x, y):
    #works only with u_net
    teacher_y_pred = teacher(x)
    student_y_pred = student(x)

    if part == 'val':
        l = loss(student_y_pred, y)
    elif part == 'train':
        l = loss(student_y_pred, teacher_y_pred, y)

    return student, student_y_pred, l

