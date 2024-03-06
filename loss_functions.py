import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F


class senti_loss(nn.Module):
    def __init__(self, senti_weight=0.25):
        super(senti_loss, self).__init__()
        self.senti_loss = smp.losses.TverskyLoss(mode='multiclass', ignore_index=11) #ignore the 11th index of plowed land
        self.unet_loss = smp.losses.TverskyLoss(mode='multiclass')
        self.senti_w = senti_weight
    def forward(self, unet_pred, senti_pred, target):
        senti_l = self.senti_loss(senti_pred, target)
        unet_l = self.unet_loss(unet_pred, target)
        combined_loss = senti_l*self.senti_w + unet_l*(1-self.senti_w)
        return combined_loss

class teacher_student_loss(nn.Module):
    def __init__(self, teacher_weight, ts_loss, rep_layer, rep_weight):
        super(teacher_student_loss, self).__init__()
        self.student_loss = smp.losses.TverskyLoss(mode='multiclass')
        if ts_loss == 'KL':
            self.teacher_loss = torch.nn.KLDivLoss(reduction='batchmean')
        elif ts_loss == 'MSE':
            self.teacher_loss = torch.nn.MSELoss()
        elif ts_loss == 'CE':
            self.teacher_loss = torch.nn.CrossEntropyLoss()
        
        self.rep_loss = torch.nn.KLDivLoss(reduction='batchmean') #fult men börjar så ändå
        self.rep_layer = rep_layer
        self.ts_loss = ts_loss
        self.teacher_w = teacher_weight
        self.rep_w = rep_weight
        
    def forward(self, student_y_pred, teacher_y_pred, target, student_last_feature=None, teacher_last_feature=None):
        student_l = self.student_loss(student_y_pred, target)
        #teacher_guess = torch.argmax(teacher_y_pred, dim=1)
        #or CE, MSE in betweem
        if self.ts_loss == 'KL':
            #log_softmax used bc KL expects log
            teacher_l = self.teacher_loss(F.log_softmax(student_y_pred, dim=1), F.softmax(teacher_y_pred/5, dim=1))/70000
        elif self.ts_loss == 'MSE':
            teacher_l = self.teacher_loss(F.softmax(student_y_pred, dim=1), F.softmax(teacher_y_pred, dim=1))*5
        elif self.ts_loss == 'CE':
            teacher_l = self.teacher_loss(F.softmax(student_y_pred, dim=1), F.softmax(teacher_y_pred, dim=1))

        if self.rep_layer:
            print(student_last_feature)
            print(teacher_last_feature)
            rep_l = self.rep_loss(F.log_softmax(student_last_feature, dim=1), F.softmax(teacher_last_feature, dim=1))

        combined_loss = teacher_l*self.teacher_w + student_l*(1-self.teacher_w) + rep_l*self.rep_w
        print('t', teacher_l.item())
        print('s', student_l.item())
        print('r', rep_l.item())

        return combined_loss
