import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.base.modules import Conv2dReLU, Attention



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

