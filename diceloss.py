import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, num_classes, epsilon=1e-5):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon

    def forward(self, input, target):
        # Convert target to one-hot encoding
        target_one_hot = F.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        # Flatten the predictions and targets
        input_flat = input.view(-1, self.num_classes)
        target_flat = target_one_hot.view(-1, self.num_classes)

        # Compute the intersection and union for each class
        intersection = torch.sum(input_flat * target_flat, dim=0)
        union = torch.sum(input_flat, dim=0) + torch.sum(target_flat, dim=0)

        # Compute the Dice loss for each class
        dice_loss_class = 1 - (2 * intersection) / (union + self.epsilon)

        # Average the Dice loss across classes
        dice_loss = torch.mean(dice_loss_class)

        return dice_loss
