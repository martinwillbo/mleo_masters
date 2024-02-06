import hydra
import numpy as np
import torch
import random
import test
from torch.utils.tensorboard import SummaryWriter
import os
from eval_model import eval_model 

@hydra.main(config_path='config', config_name='config', version_base = '1.3.2')
def main(config):
    
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)

    #NOTE: Don't have to use tensorboard to log experiments, but should implement something else if so (Aleksis have code for this).
    training_path = '/raid/dlgroupmsc/logs/2024-02-01_10-33-23' #2024-01-29_18-13-26'
    log_dir = os.path.join(training_path, 'tensorboard_test') #DON'T CHANGE, RISK OF OVERWRITING ALL PREVIOUSLY LOGGED DATA
    log_dir = log_dir  +'_' + str(config.eval_type)
    print(log_dir)
    writer = SummaryWriter(log_dir=log_dir)
    eval_type = config.eval_type
    
    #eval_model(config, writer, training_path, eval_type)
    test.eval_on_test(config, writer, training_path)

if __name__ == '__main__':
    main()