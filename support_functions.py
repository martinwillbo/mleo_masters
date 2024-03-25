import os
from unet_module import UnetFeatureMetadata, UnetFeatureMetadata_2, UnetFeatureSenti, UnetSentiDoubleLoss, UnetSentiUTAE #, UnetFeatureSentiMtd, UNetWithMetadata
import segmentation_models_pytorch as smp
#from fcnpytorch.fcn8s import FCN8s as FCN8s #smaller net!
from GAN.pix2pix import Pix2PixModel
from torchvision.models.segmentation.deeplabv3 import deeplabv3_resnet50
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence



def set_model(config, model_name, n_channels):
    if model_name == 'resnet50':
        model = deeplabv3_resnet50(weights = config.model.pretrained, progress = True, #num_classes = config.model.n_class,
                                    dim_input = n_channels, aux_loss = None, weights_backbone = config.model.pretrained_backbone)
        model.classifier[4] = nn.Conv2d(256, config.model.n_class, kernel_size=(1,1), stride=(1,1))
        model.backbone.conv1 = nn.Conv2d(n_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    #elif model_name == 'FCN8':
    #    model = FCN8s(n_class=config.model.n_class, dim_input=n_channels, weight_init='normal')
    elif model_name== 'unet':
        model = smp.Unet(
            encoder_weights="imagenet",
            encoder_name="efficientnet-b4",
            in_channels = n_channels,
            classes= config.model.n_class
        )
    #elif model_name== 'unet_mtd':
    #    model = UNetWithMetadata(n_channels=n_channels, n_class=config.model.n_class, n_metadata=6, device=config.device, reweight=config.model.reweight_late, mtd_weighting = config.model.mtd_weighting)
    elif model_name == 'unet_mtd_feature':
        model = UnetFeatureMetadata(n_channels=n_channels, n_class=config.model.n_class, n_metadata=6)
    elif model_name == 'unet_mtd_feature_2':
        model = UnetFeatureMetadata_2(n_channels=n_channels, n_class=config.model.n_class, feature_block=config.model.feature_block, linear_mtd_preprocess=config.model.linear_mtd_preprocess)
    elif config.model.name == 'unet_senti':
        model = UnetFeatureSenti(n_channels=n_channels, n_senti_channels=120, n_classes=config.model.n_class)
    elif config.model.name == 'unet_senti_double':
        model = UnetSentiDoubleLoss(n_channels=n_channels, n_senti_channels=120, n_classes=config.model.n_class)
    elif config.model.name == 'teacher_student':
        model = smp.Unet(
            encoder_weights="imagenet",
            encoder_name="efficientnet-b4",
            in_channels = config.model.teacher_student.student_channels,
            classes= config.model.n_class
        ) 
    elif config.model.name == 'unet_senti_utae':
        model = UnetSentiUTAE(n_channels=n_channels, n_senti_channels=10, n_classes=config.model.n_class)

    elif config.model.name == 'pix2pix':
        model = Pix2PixModel(config)

    return model

def get_teacher(config, teacher_path, teacher_model_type):
        teacher = set_model(config, teacher_model_type, config.model.teacher_student.teacher_channels)
        teacher_path = os.path.join(teacher_path, 'best_model.pth')
        teacher.load_state_dict(torch.load(teacher_path))
        return teacher 

def teacher_student(teacher, student, part, loss, x, y, teacher_channels, rep_layer):
    #works only with u_net

    with torch.no_grad():
        if rep_layer:
            teacher_last_feature =  teacher.encoder(x[:,-teacher_channels:, :, :])[-1]
        teacher_y_pred = teacher(x[:,-teacher_channels:, :, :])
    if rep_layer:
        student_last_feature = student.encoder(x[:,:3, :, :])[-1]   
    student_y_pred = student(x[:,:3, :, :]) #ONLY RGB!!!
    if part == 'val':
        l = loss(student_y_pred, y)
    elif part == 'train':
        if rep_layer:
            l = loss(student_y_pred, teacher_y_pred, y, student_last_feature, teacher_last_feature)
        else:
            l = loss(student_y_pred, teacher_y_pred, y)
    return student, student_y_pred, l

def collate_fn(batch):
    # Extract x_data, label_data, and time_series_data from the batch
    batch_x_data, batch_y_data, batch_time_series_data = zip(*batch)
    
    # Pad the time-series data
    padded_time_series_data = pad_sequence(batch_time_series_data, batch_first=True, padding_value=0)
    
    batch_x_data = torch.stack(batch_x_data)
    batch_y_data = torch.stack(batch_y_data)
    #padded_time_series_data = torch.stack(padded_time_series_data)
    #return torch.tensor(batch_x_data, dtype = torch.float), torch.tensor(batch_y_data, dtype = torch.long), torch.tensor(padded_time_series_data, dtype = torch.float)
    return batch_x_data, batch_y_data, padded_time_series_data





