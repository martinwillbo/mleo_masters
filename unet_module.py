import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.base.modules import Conv2dReLU, Attention
from segmentation_models_pytorch.encoders import get_encoder



class SEBlock(nn.Module):

    def __init__(self, n_channels, n_senti, reduction_ratio=16): #n_senti is sze of feature output
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Squeeze operation
        self.fc1 = nn.Linear(n_channels + n_senti, n_channels // reduction_ratio)  # First linear layer
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(n_channels // reduction_ratio, n_channels)  # Second linear layer
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, senti):
        b, c, _, _ = x.size()
        # Squeeze
        b_senti, c_senti, _, _ = senti.size()
        squeezed_features_senti = self.avg_pool(senti).view(b_senti, c_senti)
        squeezed_features = self.avg_pool(x).view(b, c)
        fused_features = torch.cat((squeezed_features, squeezed_features_senti), dim=1)
        # Recalibrate
        recalibrated_features = self.relu(self.fc1(fused_features))
        recalibrated_features = self.sigmoid(self.fc2(recalibrated_features))
        # Apply recalibration to original feature maps
        x = x * recalibrated_features.view(b, c, 1, 1)
        return x
    
class UnetFeatureMetadata_2(nn.Module):

    def __init__(self, n_channels, n_class, feature_block=1, linear_mtd_preprocess=False):
        self.feature_block = feature_block
        feature_channel_list = [5, 48, 32, 56, 160, 448]
        super(UnetFeatureMetadata_2, self).__init__()
        if linear_mtd_preprocess:
            self.SEBlock = SEBlockLinear(feature_channel_list[feature_block], 6)
        else:
            self.SEBlock = SEBlockWithMetadata(feature_channel_list[feature_block], 6)
        self.unet = smp.Unet(
            encoder_weights="imagenet",
            encoder_name="efficientnet-b4",
            in_channels = n_channels,
            classes= n_class
        )  # Your existing U-Net model

    def forward(self, x, mtd):
        #get features from encoder
        features = self.unet.encoder(x)
        #for f in features:
        #    print(f.shape)
        #print(features[1].shape)
        features[self.feature_block] = self.SEBlock(features[self.feature_block], mtd)
        #print(self.unet.decoder.blocks[4])
        #print(self.unet.segmentation_head)
        #input features to decoder
        y_pred = self.unet.decoder(*features)
        y_pred = self.unet.segmentation_head(y_pred)
        return y_pred
    

    
class UnetFeatureSenti(nn.Module):

    def __init__(self, n_channels, n_senti_channels, n_classes):
        super(UnetFeatureSenti, self).__init__()
        self.unet = smp.Unet(
            encoder_weights="imagenet",
            encoder_name="efficientnet-b4",
            in_channels = n_channels,
            classes= n_classes
        )  # Your existing U-Net model
        #overwrite
        self.senti_encoder = get_encoder('efficientnet-b0', weights = 'imagenet', encoder_depth=5, in_channels=n_senti_channels)
        feature_channel_list = [5, 48, 32, 56, 160, 448]
        feature_senti_channel_list = [120, 32, 24, 40, 112, 320]

        self.SEBlock_1 = SEBlock(feature_channel_list[1], feature_senti_channel_list[1])
        SEBlock_2 = SEBlock(feature_channel_list[2], feature_senti_channel_list[2])
        SEBlock_3 = SEBlock(feature_channel_list[3], feature_senti_channel_list[3])
        SEBlock_4 = SEBlock(feature_channel_list[4], feature_senti_channel_list[4])
        SEBlock_5 = SEBlock(feature_channel_list[5], feature_senti_channel_list[5])

        self.SEBlock_list = [self.SEBlock_1, SEBlock_2, SEBlock_3, SEBlock_4, SEBlock_5]

    def forward(self, x, senti):
        #print(senti.shape)
        
        senti = senti.view(senti.shape[0], -1 , senti.shape[-2], senti.shape[-1])
        #print(senti.shape)
        #get features from encoder
        features = self.unet.encoder(x)
        #get features from senti_encoder
        features_senti = self.senti_encoder(senti)
        
        #SE_features = 0*features.copy()
        #SE_features[0] = features[0]
        features[1] = self.SEBlock_1(features[1], features_senti[1])
        

        for i in range(1,6):
            features[i] = self.SEBlock_list[i](features[i], features_senti[i])

        y_pred = self.unet.decoder(*features)
        y_pred = self.unet.segmentation_head(y_pred)
        
        return y_pred




class UnetFeatureMetadata(nn.Module):

    def __init__(self, n_channels, n_class, n_metadata):
        super(UnetFeatureMetadata, self).__init__()
        self.unet = smp.Unet(
            encoder_weights="imagenet",
            encoder_name="efficientnet-b4",
            in_channels = n_channels,
            classes= n_class
        )  # Your existing U-Net model
        #overwrite
        self.unet.decoder.blocks[0].conv1 = Conv2dReLU(614, 256, kernel_size=3, padding=1, use_batchnorm=True)
        self.unet.decoder.blocks[0].attention1= Attention(None, in_channels=614)

    def forward(self, x, mtd):
        #get features from encoder
        features = self.unet.encoder(x)
        #for f in features:
        #    print(f.shape)
        #initially shape 8, 6 -> 8,6,16,16 (as in 5th feature of 8, 448, 16, 16)
        mtd = mtd.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 16, 16)
        features[5] = torch.cat((features[5], mtd), dim=1)
        #print(self.unet.decoder.blocks[4])
        #print(self.unet.segmentation_head)
        #input features to decoder
        y_pred = self.unet.decoder(*features)
        y_pred = self.unet.segmentation_head(y_pred)
        return y_pred