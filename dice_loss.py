import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, num_classes, epsilon=1e-5):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon

    def forward(self, y_pred, y):
        print(y_pred.shape)
        print(y.shape)

        dice_coeffs = torch.zeros((self.num_classes,), dtype=torch.float)  #creates empty vecn
        
        y_pred_flat = y_pred.view(-1)
        y_flat = y.view(-1)

        for i in range(self.num_classes): #for all classes

            y_flat_i = y_flat == i #sets ones where y_flat is equal to i
            pred_flat_i = y_pred_flat == i
            intersection_i = torch.logical_and(y_flat_i, pred_flat_i) #where they match
            union_i = torch.logical_or(y_flat_i, pred_flat_i) #everything together
            num_intersection_i = torch.count_nonzero(intersection_i).item() #how big is the intersection
            num_union_i = torch.count_nonzero(union_i).item() #how big is the union

            dice_coeffs[i] = (2 * num_intersection_i + self.epsilon)/(num_union_i + self.epsilon)

        dice_loss = 1 - torch.mean(dice_coeffs)
        print(dice_coeffs)
        return dice_loss