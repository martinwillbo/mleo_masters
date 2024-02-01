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

        y_pred = F.softmax(y_pred, dim=1)
        
        # One-hot encode the y
        y_onehot = F.one_hot(y, num_classes=self.config.model.n_class).permute(0, 3, 1, 2).float()

        print(y_pred[0,5,:,:])
        print(y_onehot[0,5,:,:])
        
        # Calculate intersection and union for each class
        intersection = torch.sum(y_pred * y_onehot, dim=(2, 3))
        cardinality = torch.sum(y_pred + y_onehot, dim=(2, 3))
        
        # Calculate Dice coefficient for each class
        dice_coeff = (2. * intersection) / (cardinality + self.epsilon)
        
        # Calculate 1 - Dice for the loss (to be minimized)
        loss = 1 - torch.mean(dice_coeff)
        #print(loss.dtype) float32
        return loss

    def dontuse(self, y_pred, y):
        y = y.to(torch.float32)
        y_pred = y_pred.to(torch.float32)

        y_pred = torch.softmax(y_pred, dim=1)

        dice_coeffs = torch.zeros((self.num_classes,), dtype=torch.float).to(self.config.device)  #creates empty vecn
        
        y_pred_flat = y_pred.view(-1)
        y_flat = y.view(-1)

        for i in range(self.num_classes): #for all classes

            y_flat_i = y_flat == i #sets ones where y_flat is equal to i
            y_pred_flat_i = y_pred_flat == i
            intersection_i = torch.logical_and(y_flat_i, y_pred_flat_i) #where they match
            num_intersection_i = torch.count_nonzero(intersection_i).item() #how big is the intersection
            num_y_i = torch.count_nonzero(y_flat_i).item() #how big is the union
            num_y_pred_i = torch.count_nonzero(y_pred_flat_i).item()

            dice_coeffs[i] = (2 * num_intersection_i + self.epsilon)/(num_y_i + num_y_pred_i + self.epsilon)

        
        dice_loss = torch.tensor(1.0) - torch.mean(dice_coeffs)
        dice_loss.requires_grad = True # har printat dtype och den är float32     
        
        return dice_loss 