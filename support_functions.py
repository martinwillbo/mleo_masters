from unet_module import UnetFeatureMetadata, UnetFeatureMetadata_2, UnetFeatureSenti, UnetSentiDoubleLoss #, UnetFeatureSentiMtd, UNetWithMetadata
import segmentation_models_pytorch as smp
#from fcnpytorch.fcn8s import FCN8s as FCN8s #smaller net!
from torchvision.models.segmentation.deeplabv3 import deeplabv3_resnet50
import torch
import torch.nn as nn

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
            in_channels = config.model.n_channels,
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
    #elif config.model.name == 'unet_senti_mtd':
    #    model = UnetFeatureSentiMtd(n_channels=n_channels, n_senti_channels=120, n_metadata=6, n_classes=config.model.n_class, w=config.model.mtd_weighting)
    return model