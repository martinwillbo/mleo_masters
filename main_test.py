import hydra
import numpy as np
import torch
import random
import test
from torch.utils.tensorboard import SummaryWriter
import os

@hydra.main(config_path='config', config_name='config', version_base = '1.3.2')
def main(config):
    
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)

    #NOTE: Don't have to use tensorboard to log experiments, but should implement something else if so (Aleksis have code for this).
    training_path = '/PATH WHERE MODEL IS'
    log_dir = os.path.join(training_path, 'tensorboard_test')
    writer = SummaryWriter(log_dir=log_dir)
    test.eval_on_test(config, writer, training_path)

if __name__ == '__main__':
    main()