import torch
import torch.nn as nn
import torchvision.models as models

class DeepLabV3Modified(nn.Module):
    def __init__(self, config):

        num_classes = config.model.n_class
        num_channels = config.model.n_channels

        super(DeepLabV3Modified, self).__init__()

        # Load the pre-trained DeepLabV3 model with ResNet-50 backbone
        self.resnet50 = models.segmentation.deeplabv3_resnet50(pretrained=True)

        # Modify the input channels of the first convolution layer
        self.resnet50.backbone.conv1 = nn.Conv2d(num_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # Change the number of input channels for the ASPP module
        self.resnet50.classifier[0] = nn.Conv2d(256 * 4, num_classes, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        return self.resnet50(x)