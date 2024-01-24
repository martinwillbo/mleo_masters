import torch
import torch.nn as nn
import torchvision.models as models

class DeepLabV3_ResNet50(nn.Module):
    def __init__(self, config):

        num_classes = config.model.n_class
        num_channels = config.model.n_channels

        super(DeepLabV3_ResNet50, self).__init__()

        # Load the pre-trained ResNet-50 model from torchvision
        resnet50 = models.resnet50(weights = config.model.pretrained_backbone)

        # Modify the first layer to accept 5 channels
        resnet50.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Extract the features from the modified ResNet-50 backbone
        self.features = nn.Sequential(
            resnet50.conv1,
            resnet50.bn1,
            resnet50.relu,
            resnet50.maxpool,
            resnet50.layer1,
            resnet50.layer2,
            resnet50.layer3,
            resnet50.layer4
        )

        # Define the ASPP module
        self.aspp = ASPP(in_channels=2048, out_channels=256, rates=[6, 12, 18, 24])

        # Define the final segmentation head
        self.segmentation_head = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        # Forward pass through the modified ResNet-50 backbone
        x = self.features(x)

        # Forward pass through the ASPP module
        x = self.aspp(x)

        # Forward pass through the segmentation head
        x = self.segmentation_head(x)

        return x

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, rates):
        super(ASPP, self).__init__()

        # Atrous Spatial Pyramid Pooling (ASPP) module
        self.conv1x1_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv1x1_2 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv1x1_3 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv1x1_4 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv1x1_5 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.atrous_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rates[0], dilation=rates[0])
        self.atrous_conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rates[1], dilation=rates[1])
        self.atrous_conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rates[2], dilation=rates[2])
        self.atrous_conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rates[3], dilation=rates[3])

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1x1_6 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.relu = nn.ReLU()

    def forward(self, x):
        # Forward pass through each branch of ASPP
        x1 = self.conv1x1_1(x)
        x2 = self.atrous_conv1(x)
        x3 = self.atrous_conv2(x)
        x4 = self.atrous_conv3(x)
        x5 = self.atrous_conv4(x)
        x6 = self.global_avg_pool(x)
        x6 = self.conv1x1_6(x6)
        x6 = nn.functional.interpolate(x6, size=x.size()[2:], mode='bilinear', align_corners=False)

        # Concatenate the results from all branches
        out = torch.cat((x1, x2, x3, x4, x5, x6), dim=1)

        return out