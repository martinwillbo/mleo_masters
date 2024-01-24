import torch
import torch.nn as nn
import torchvision.models as models

class DeepLabV3Modified(nn.Module):
    def __init__(self, config):

        num_classes = config.model.n_class
        num_channels = config.model.n_channels

        super(DeepLabV3Modified, self).__init__()

        # Load the pre-trained DeepLabV3 model with ResNet-50 backbone
        self.resnet50 = models.segmentation.deeplabv3_resnet50(weights = config.model.pretrained, weights_backbone = config.model.pretrained_backbone)

        # Modify the input channels of the first convolution layer
        self.resnet50.backbone.conv1 = nn.Conv2d(num_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        aspp_in_channels = self.resnet50.classifier[0].in_channels # 256*4 acc to chatgpt

        # Change the number of input channels for the ASPP module
        self.resnet50.classifier[0] = nn.Conv2d(aspp_in_channels, num_classes, kernel_size=(1, 1), stride=(1, 1)) #was 256*4 from the begininin

    def forward(self, x):
        return self.resnet50(x)