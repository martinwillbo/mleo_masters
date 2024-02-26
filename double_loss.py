import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
class senti_loss(nn.Module):
    def __init__(self, senti_weight=0.15):
        super(senti_loss, self).__init__()
        self.senti_loss = smp.losses.TverskyLoss(mode='multiclass', ignore_index=11) #ignore the 11th index of plowed land
        self.unet_loss = smp.losses.TverskyLoss(mode='multiclass')
        self.senti_w = senti_weight
    def forward(self, unet_pred, senti_pred, target):
        senti_l = self.senti_loss(senti_pred, target)
        unet_l = self.unet_loss(unet_pred, target)
        combined_loss = senti_l*self.senti_w + unet_l*(1-self.senti_w)
        return combined_loss