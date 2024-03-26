import hydra
import numpy as np
import torch
import random
import test
from eval_model import eval_model 
from torch.utils.tensorboard import SummaryWriter
import os
@hydra.main(config_path='config', config_name='config', version_base = '1.3.2')
def main(config):
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)

    #NOTE: Don't have to use tensorboard to log experiments, but should implement something else if so (Aleksis has code for this).
    training_path = '../log_res/best_model_5_channels_5'# 2024-02-08_17-09-25/'
    #log_dir = os.path.join(training_path, 'tensorboard')
    #log_dir = log_dir  +'_' + str(config.eval_type)
    #writer = SummaryWriter(log_dir=log_dir)
    writer=None

    eval_model(config, writer, training_path, config.eval_type)
    #test.eval_on_test(config, writer, training_path)

if __name__ == '__main__':
    main()