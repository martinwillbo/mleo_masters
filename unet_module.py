import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.base.modules import Conv2dReLU, Attention
from segmentation_models_pytorch.encoders import get_encoder



class AdditionalHead(nn.Module):
    def __init__(self, n_metadata, n_class):
        super(AdditionalHead, self).__init__()
        # Define your layers here, e.g., fully connected layers
        self.fc1 = nn.Linear(n_metadata, 128)  # Example layer
        self.fc2 = nn.Linear(128, 64)  # Example layer
        #self.fc3 = nn.Linear(128, 64) #added, updated sizes used 256, 128, 64
        self.fc4 = nn.Linear(64, n_class)  # Output layer for 19 classes

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        x = self.fc4(x)
        output = F.softmax(x, dim=1)
        return output

class UNetWithMetadata(nn.Module):
    def __init__(self, n_channels, n_class, n_metadata, device, reweight, mtd_weighting):
        self.w = mtd_weighting
        self.reweight=reweight
        super(UNetWithMetadata, self).__init__()
        self.unet = smp.Unet(
            encoder_weights="imagenet",
            encoder_name="efficientnet-b4",
            in_channels = n_channels,
            classes= n_class, 
        )  # Your existing U-Net model
        self.additional_head = AdditionalHead(n_metadata, n_class)  # The additional head you defined
        class_freq = torch.tensor([8.14, 8.25, 13.72, 3.47, 4.88, 2.74, 15.38, 6.95, 3.13, 17.84, 10.98, 3.88, 0.01, 0.15, 0.15, 0.05, 0.01, 0.12, 0.14])
        class_freq /= 100
        class_freq = class_freq.to(device)
        self.class_freq = class_freq


    def forward(self, x, mtd):
        # Forward pass through the original U-Net
        unet_output = self.unet(x)  # Shape: [batch_size, 19, 512, 512]

        # Forward pass through the additional head
        mtd_output = self.additional_head(mtd)  # Shape: [batch_size, 19]
        #gives high probabilities to normal classes - reasonable, we want to show how much more likely

        if self.reweight:
            mtd_output = mtd_output/self.class_freq
         
        mtd_output = mtd_output.unsqueeze(-1).unsqueeze(-1)  # Shape: [batch_size, 19, 1, 1]
        if self.reweight:
            reweighted_pred = mtd_output*unet_output
            # Optionally expand to match the U-Net output shape
            reweighted_pred_expanded = reweighted_pred.expand(-1, -1, 512, 512)
            combined_output = unet_output*(1-self.w) + torch.softmax(reweighted_pred_expanded, dim=1)*self.w

        else:
            mtd_expanded = mtd_output.expand(-1, -1, 512, 512)        
            combined_output = unet_output*(1-self.w) + mtd_expanded*self.w #(w=0.3 > w=0.1)
        #do this mult before expansion? 

        return combined_output

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
    

class SEBlockWithMetadata(nn.Module):
    def __init__(self, n_channels, n_metadata, reduction_ratio=16):
        super(SEBlockWithMetadata, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Squeeze operation
        self.fc1 = nn.Linear(n_channels + n_metadata, n_channels // reduction_ratio)  # First linear layer
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(n_channels // reduction_ratio, n_channels)  # Second linear layer
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, metadata):
        b, c, _, _ = x.size()
        # Squeeze
        squeezed_features = self.avg_pool(x).view(b, c)
        fused_features = torch.cat((squeezed_features, metadata), dim=1)
        # Recalibrate
        recalibrated_features = self.relu(self.fc1(fused_features))
        recalibrated_features = self.sigmoid(self.fc2(recalibrated_features))
        
        # Apply recalibration to original feature maps
        x = x * recalibrated_features.view(b, c, 1, 1)
        
        return x
    

class SEBlockLinear(nn.Module):
    def __init__(self, n_channels, n_metadata, reduction_ratio=8):
        super(SEBlockLinear, self).__init__()
        self.metadata_processor = nn.Sequential(
            nn.Linear(n_metadata, 64),  # Example transformation to an intermediate size
            nn.ReLU(),
            nn.Linear(64, n_channels),  # Match the dimensionality to feature maps' channels
            #nn.ReLU() #seems reasonable? was not in before
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(n_channels * 2, n_channels // reduction_ratio)  # Adjust for concatenated size
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(n_channels // reduction_ratio, n_channels)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, metadata):
        b, c, _, _ = x.size()
        # Process metadata
        processed_metadata = self.metadata_processor(metadata)
        
        # Squeeze and fuse
        squeezed_features = self.avg_pool(x).view(b, c)
        fused_features = torch.cat((squeezed_features, processed_metadata), dim=1)
        
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


class SEBlock(nn.Module):

    def __init__(self, n_channels, n_senti, reduction_ratio=2): #n_senti is sze of feature output
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
        feature_senti_channel_list = [120, 32, 24, 40, 112, 320] #-b0
        #feature_senti_channel_list = [120, 16, 24, 32, 96, 1280]

        self.SEBlock_1 = SEBlock(feature_channel_list[1], feature_senti_channel_list[1])
        self.SEBlock_2 = SEBlock(feature_channel_list[2], feature_senti_channel_list[2])
        self.SEBlock_3 = SEBlock(feature_channel_list[3], feature_senti_channel_list[3])
        self.SEBlock_4 = SEBlock(feature_channel_list[4], feature_senti_channel_list[4])
        self.SEBlock_5 = SEBlock(feature_channel_list[5], feature_senti_channel_list[5])

#        self.SEBlock_list = [self.SEBlock_1, SEBlock_2, SEBlock_3, SEBlock_4, SEBlock_5]

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
        #features[2] = self.SEBlock_2(features[2], features_senti[2])
        #features[3] = self.SEBlock_3(features[3], features_senti[3])
        #features[4] = self.SEBlock_4(features[4], features_senti[4])
        #features[5] = self.SEBlock_5(features[5], features_senti[5])
        

       # for i in range(1,6):
       #     features[i] = self.SEBlock_list[i](features[i], features_senti[i])

        y_pred = self.unet.decoder(*features)
        y_pred = self.unet.segmentation_head(y_pred)
        
        return y_pred
    

class UnetFeatureSentiMtd(nn.Module):

    def __init__(self, n_channels, n_senti_channels, n_metadata, n_classes, w):
        super(UnetFeatureSentiMtd, self).__init__()
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
        self.w = w

        self.SEBlock_1 = SEBlock(feature_channel_list[1], feature_senti_channel_list[1])
        self.SEBlock_2 = SEBlock(feature_channel_list[2], feature_senti_channel_list[2])
        self.SEBlock_3 = SEBlock(feature_channel_list[3], feature_senti_channel_list[3])
        self.SEBlock_4 = SEBlock(feature_channel_list[4], feature_senti_channel_list[4])
        self.SEBlock_5 = SEBlock(feature_channel_list[5], feature_senti_channel_list[5])

        self.additional_head = AdditionalHead(n_metadata, n_classes)  # The additional head you defined

#        self.SEBlock_list = [self.SEBlock_1, SEBlock_2, SEBlock_3, SEBlock_4, SEBlock_5]

    def forward(self, x, senti, mtd):
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
        features[2] = self.SEBlock_2(features[2], features_senti[2])
        features[3] = self.SEBlock_3(features[3], features_senti[3])
        features[4] = self.SEBlock_4(features[4], features_senti[4])
        features[5] = self.SEBlock_5(features[5], features_senti[5])
        

       # for i in range(1,6):
       #     features[i] = self.SEBlock_list[i](features[i], features_senti[i])

        y_pred = self.unet.decoder(*features)
        y_pred = self.unet.segmentation_head(y_pred)

        mtd_output = self.additional_head(mtd)
        mtd_output = mtd_output.unsqueeze(-1).unsqueeze(-1)
        mtd_expanded = mtd_output.expand(-1, -1, 512, 512)     
        combined_output = y_pred*(1-self.w) + mtd_expanded*self.w #(w=0.3 > w=0.1)
        
        return combined_output
    

class UnetSentiDoubleLoss(nn.Module):
    def __init__(self, n_channels, n_senti_channels, n_classes):
        super(UnetSentiDoubleLoss, self).__init__()
        self.unet = smp.Unet(
            encoder_weights="imagenet",
            encoder_name="efficientnet-b4",
            in_channels = n_channels,
            classes= n_classes
        )  # Your existing U-Net model
        self.unet_senti = smp.Unet(
            encoder_weights="imagenet",
            encoder_name="efficientnet-b0",
            in_channels = n_senti_channels,
            classes= n_classes
        )
        #overwrite
        feature_channel_list = [5, 48, 32, 56, 160, 448]
        feature_senti_channel_list = [120, 16, 16, 16, 16, 16]
        self.SEBlock_1 = SEBlock(feature_channel_list[1], feature_senti_channel_list[1])
        self.SEBlock_2 = SEBlock(feature_channel_list[2], feature_senti_channel_list[2])
        self.SEBlock_3 = SEBlock(feature_channel_list[3], feature_senti_channel_list[3])
        self.SEBlock_4 = SEBlock(feature_channel_list[4], feature_senti_channel_list[4])
        self.SEBlock_5 = SEBlock(feature_channel_list[5], feature_senti_channel_list[5])
        self.reshape_senti_output = nn.Sequential(nn.Upsample(size=(512,512), mode='nearest'),
                                                nn.Conv2d(16, n_classes, 1)
                                                )
    def forward(self, x, senti):
        senti = senti.view(senti.shape[0], -1 , senti.shape[-2], senti.shape[-1])
        #get features from encoder
        features = self.unet.encoder(x)
        #get features from senti_encoder
        features_senti = self.unet_senti.encoder(senti)
        #pass senti through decoder
        decoded_senti = self.unet_senti.decoder(*features_senti)
        # squeeze and excite decoded senti and encoded aerial
        features[1] = self.SEBlock_1(features[1], decoded_senti)
        features[2] = self.SEBlock_2(features[2], decoded_senti)
        features[3] = self.SEBlock_3(features[3], decoded_senti)
        features[4] = self.SEBlock_4(features[4], decoded_senti)
        features[5] = self.SEBlock_5(features[5], decoded_senti)
        #predict with senti
        transform = T.CenterCrop((10, 10))
        senti_out = transform(decoded_senti)
        senti_pred_out = self.reshape_senti_output(senti_out)
        #predict using aerial
        decoded = self.unet.decoder(*features)
        y_pred = self.unet.segmentation_head(decoded)
        return y_pred, senti_pred_out
    

class UnetPredictPriv(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UnetPredictPriv, self).__init__()
        self.unet = smp.Unet(
            encoder_weights="imagenet",
            encoder_name="efficientnet-b4",
            in_channels = n_channels,
            classes= n_classes
        )
        self.NIR_decoder = smp.Unet(
            encoder_weights="imagenet",
            encoder_name="efficientnet-b4",
            in_channels = n_channels,
            classes= 1 #one scalar per priv channel
        ).decoder
        self.NIR_segmentation_head = smp.Unet(
            encoder_weights="imagenet",
            encoder_name="efficientnet-b4",
            in_channels = n_channels,
            classes= 1 #entire eight bit
        ).segmentation_head

        self.elev_decoder = smp.Unet(
            encoder_weights="imagenet",
            encoder_name="efficientnet-b4",
            in_channels = n_channels,
            classes= 1 #one scalar per priv channel
        ).decoder
        self.elev_segmentation_head = smp.Unet(
            encoder_weights="imagenet",
            encoder_name="efficientnet-b4",
            in_channels = n_channels,
            classes= 1 #entire eight bit
        ).segmentation_head


    def forward(self, x):
        features = self.unet.encoder(x[:,:3,:,:])

        unet_decoded = self.unet.decoder(*features)
        y_pred = self.unet.segmentation_head(unet_decoded)

        NIR_decoded = self.NIR_decoder(*features)
        NIR_priv_pred = self.NIR_segmentation_head(NIR_decoded)

        elev_decoded = self.elev_decoder(*features)
        elev_priv_pred = self.elev_segmentation_head(elev_decoded)

        y_priv_pred = torch.cat((NIR_priv_pred, elev_priv_pred), dim=1)

        return y_pred, y_priv_pred

