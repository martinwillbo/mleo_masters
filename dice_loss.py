import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, config, epsilon=1e-5):
        super(DiceLoss, self).__init__()
        self.num_classes = config.model.n_class
        self.epsilon = epsilon
        self.config = config

    def forward(self, y_pred, y):
        #softmax
        y_pred = F.softmax(y_pred, dim=1)
        
        # One-hot encode the y
        y_onehot = F.one_hot(y, num_classes=self.config.model.n_class).permute(0, 3, 1, 2).float()
        
        # Calculate intersection and union for each class
        intersection = torch.sum(y_pred * y_onehot, dim=(2, 3))
        cardinality = torch.sum(y_pred + y_onehot, dim=(2, 3))
        
        # Calculate Dice coefficient for each class
        dice_coeff = (2 * intersection) / (cardinality + self.epsilon)

        # 1-12 get twice the weight as 13-19
        #weights_12 = 2/31*torch.ones(12)
        #weights_13_plus = 1/32*torch.ones(7)
        #weights = torch.cat((weights_12, weights_13_plus)).cuda()

        #dice_weighted = dice_coeff # * weights        

        # weigted average over classes
        loss = 1 - torch.mean(dice_coeff)
        #loss.to('cuda:0')

        return loss