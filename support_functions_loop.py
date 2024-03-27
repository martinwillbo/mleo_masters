from unet_module import UnetPredictPriv, UnetPrivForward, UNetWithMetadata, UnetFeatureMetadata, UnetFeatureMetadata_2, UnetFeatureSenti, UnetSentiDoubleLoss, UnetFeatureSentiMtd
import segmentation_models_pytorch as smp
from fcnpytorch.fcn8s import FCN8s as FCN8s #smaller net!
from torchvision.models.segmentation.deeplabv3 import deeplabv3_resnet50
import torch
import torch.nn as nn
import os

from custom_losses import CE_tversky_focal_loss,CE_tversky_loss, senti_loss, teacher_student_loss, multi_teacher_loss, avg_teacher_loss, predict_priv_loss, generate_priv_loss


def set_model(config, model_name, n_channels):
    if model_name == 'resnet50':
        model = deeplabv3_resnet50(weights = config.model.resnet50.pretrained, progress = True, #num_classes = config.model.n_class,
                                    dim_input = n_channels, aux_loss = None, weights_backbone = config.model.resnet50.pretrained_backbone)  
        model.classifier[4] = nn.Conv2d(256, config.model.n_class, kernel_size=(1,1), stride=(1,1))    
        model.backbone.conv1 = nn.Conv2d(n_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    
    elif model_name == 'FCN8':
        model = FCN8s(n_class=config.model.n_class, dim_input=n_channels, weight_init='normal')

    elif model_name== 'unet':
        model = smp.Unet(
            encoder_weights="imagenet",
            encoder_name="efficientnet-b4",
            in_channels = n_channels,
            classes= config.model.n_class
        )

    elif model_name== 'unet_mtd':
        model = UNetWithMetadata(n_channels=n_channels, n_class=config.model.n_class, n_metadata=6, device=config.device, reweight=config.model.mtd.reweight_late, mtd_weighting = config.model.mtd.mtd_weighting)
    
    elif model_name == 'unet_mtd_feature':
        model = UnetFeatureMetadata(n_channels=n_channels, n_class=config.model.n_class, n_metadata=6)
    
    elif model_name == 'unet_mtd_feature_2':
        model = UnetFeatureMetadata_2(n_channels=n_channels, n_class=config.model.n_class, feature_block=config.model.mtd.feature_block, linear_mtd_preprocess=config.model.mtd.linear_mtd_preprocess)
    
    elif model_name == 'unet_senti':
        model = UnetFeatureSenti(n_channels=n_channels, n_senti_channels=120, n_classes=config.model.n_class)

    elif model_name == 'unet_senti_double':
        model = UnetSentiDoubleLoss(n_channels=n_channels, n_senti_channels=120, n_classes=config.model.n_class)

    elif model_name == 'unet_senti_mtd':
        model = UnetFeatureSentiMtd(n_channels=n_channels, n_senti_channels=120, n_metadata=6, n_classes=config.model.n_class, w=config.model.mtd.mtd_weighting)

    elif model_name == 'unet_predict_priv':
        model = UnetPredictPriv(n_channels=n_channels, n_classes=config.model.n_class)

    elif model_name == 'unet_generate_priv':
        model = model = smp.Unet(
            encoder_weights="imagenet",
            encoder_name="efficientnet-b4",
            in_channels = n_channels,
            classes= config.model.n_class
        )
    elif model_name == 'unet_priv_forward':
        model = UnetPrivForward(n_classes=config.model.n_class)

    return model

def set_loss(loss_function, config):
    if loss_function == 'CE':
        train_loss = nn.CrossEntropyLoss()
        eval_loss = smp.losses.TverskyLoss(mode='multiclass')
    elif loss_function == 'tversky':
        train_loss = smp.losses.TverskyLoss(mode='multiclass')
        eval_loss = smp.losses.TverskyLoss(mode='multiclass')
    elif loss_function == 'CE_tversky':
        train_loss = CE_tversky_loss(ce_weight=config.loss_w.ce_w)
        eval_loss = smp.losses.TverskyLoss(mode='multiclass')
    elif loss_function == 'CE_tversky_focal':
        train_loss = CE_tversky_focal_loss(ce_weight=config.loss_w.ce_w, focal_weight=config.loss_w.focal_w)
        eval_loss = smp.losses.TverskyLoss(mode='multiclass')
    elif loss_function == 'senti_loss':
        train_loss = senti_loss()
        eval_loss = senti_loss()
    elif loss_function == 'teacher_student_loss':
        train_loss = teacher_student_loss(teacher_weight=config.model.teacher_student.alpha, 
                                          ts_loss=config.model.teacher_student.ts_loss, 
                                          student_T=config.model.teacher_student.student_T,
                                          teacher_T=config.model.teacher_student.teacher_T,
                                          R=config.model.teacher_student.R,
                                          student_loss=config.model.teacher_student.student_loss)
        eval_loss = smp.losses.TverskyLoss(mode='multiclass') #let eval loss be for only student
    elif loss_function == 'multi_teacher_loss':
        train_loss = multi_teacher_loss(teacher_weight=config.model.teacher_student.alpha, 
                                          ts_loss=config.model.teacher_student.ts_loss, 
                                          student_T=config.model.teacher_student.student_T,
                                          teacher_T=config.model.teacher_student.teacher_T,
                                          R=config.model.teacher_student.R)
        eval_loss = smp.losses.TverskyLoss(mode='multiclass')
    
    elif loss_function == 'avg_teacher_loss':
        train_loss = avg_teacher_loss(teacher_weight=config.model.teacher_student.alpha, 
                                          ts_loss=config.model.teacher_student.ts_loss, 
                                          student_T=config.model.teacher_student.student_T,
                                          teacher_T=config.model.teacher_student.teacher_T,
                                          R=config.model.teacher_student.R)
        eval_loss = smp.losses.TverskyLoss(mode='multiclass')
    
    elif loss_function == 'predict_priv_loss':
        train_loss = predict_priv_loss(config.dataset.mean[3:], config.dataset.std[3:])
        eval_loss = smp.losses.TverskyLoss(mode='multiclass')

    elif loss_function == 'generate_priv_loss':
        model_train_loss = smp.losses.TverskyLoss(mode='multiclass')
        priv_train_loss = generate_priv_loss()
        model_eval_loss = smp.losses.TverskyLoss(mode='multiclass')
        priv_eval_loss = generate_priv_loss()
        return model_train_loss, priv_train_loss, model_eval_loss, priv_eval_loss

    elif loss_function =='priv_forward_loss':
        train_loss = generate_priv_loss()
        eval_loss = smp.losses.TverskyLoss(mode='multiclass')
        
    return train_loss, eval_loss

def get_loss_y_pred(model_name, loss_function, loss, model, x, mtd, senti, y):
    if model_name == 'resnet50':
        y_pred = model(x)['out'] #NOTE: dlv3_r50 returns a dictionary
    elif model_name == 'FCN8' or model_name == 'unet':
        y_pred = model(x)
    elif model_name == 'unet_mtd' or model_name == 'unet_mtd_feature' or model_name == 'unet_mtd_feature_2':
        y_pred = model(x, mtd)
    elif model_name== 'unet_senti':
        y_pred = model(x,senti)
    elif model_name == 'unet_senti_mtd':
        y_pred = model(x, senti, mtd)
    elif model_name == 'unet_senti_double':
        y_pred, y_pred_senti = model(x, senti)

    if loss_function == 'senti_loss':
        l = loss(y_pred, y_pred_senti, y)
    else:
        l = loss(y_pred, y)

    return model, y_pred, l

def get_teacher(config, teacher_path, teacher_channels, teacher_model_type='unet') :
    teacher = set_model(config, teacher_model_type, teacher_channels)
    teacher_path = os.path.join(teacher_path, 'best_model.pth')
    teacher.load_state_dict(torch.load(teacher_path))
    return teacher


def teacher_student(teacher, student, part, loss, x, y, student_spec_channels, teacher_spec_channels):
    #works only with u_net
    with torch.no_grad():
        teacher_y_pred = teacher(x[:,teacher_spec_channels, :, :])
    student_y_pred = student(x[:,student_spec_channels, :, :]) #ONLY RGB!!!
    
    if part == 'val':
        l = loss(student_y_pred, y)
    elif part == 'train':
        l = loss(student_y_pred, teacher_y_pred, y)

    return student, student_y_pred, l

def multi_teacher(teacher_1, teacher_2, student, part, loss, x, y, student_spec_channels, teacher_1_spec_channels, teacher_2_spec_channels):
    student_y_pred = student(x[:,student_spec_channels,:,:])
    if part == 'val':
        l = loss(student_y_pred, y)
        return student, student_y_pred, l
    
    with torch.no_grad():
        teacher_1_y_pred = teacher_1(x[:, teacher_1_spec_channels, :, :])
        teacher_2_y_pred = teacher_2(x[:, teacher_2_spec_channels, :, :])
    
    l = loss(student_y_pred, teacher_1_y_pred, teacher_2_y_pred, y)

    return student, student_y_pred, l

def generate_priv(priv_generator, model, priv_generator_loss, model_loss, x, y):
    generated_priv, y_pred_1 = priv_generator(x[:, :3, :, :])
    detached_gen_priv = generated_priv.detach()
    all_channels = torch.cat( (x[:,:3,:,:],detached_gen_priv), dim=1)
    if torch.isnan(x).any():
        print("NaN detected in input")

    # After forward pass
    if torch.isnan(generated_priv).any():
        print("NaN detected in output")

    y_pred = model(all_channels)

    model_l = model_loss(y_pred, y)

    generated_priv_l, first_generated_priv_l, priv_loss_mean, priv_loss_std = priv_generator_loss(y_pred_1, generated_priv, y, x[:,3:,:,:])

    return model, priv_generator, y_pred, model_l, generated_priv_l, first_generated_priv_l, priv_loss_mean, priv_loss_std, generated_priv

def priv_forward(model, train_loss, eval_loss, part, x, y):
    y_pred, generated_priv = model(x[:,:3,:,:])
    if part == 'train':
        l, first_priv_loss, priv_loss_mean, priv_loss_std = train_loss(y_pred, generated_priv, y, x[:,3:,:,:])
        return model, y_pred, l, first_priv_loss, priv_loss_mean, priv_loss_std, generated_priv
    elif part =='val':
        l = eval_loss(y_pred,y)
        generated_priv_l, first_priv_loss, priv_loss_mean, priv_loss_std = train_loss(y_pred, generated_priv, y, x[:,3:,:,:])
        return model, y_pred, l, generated_priv_l, first_priv_loss, priv_loss_mean, priv_loss_std, generated_priv

   





