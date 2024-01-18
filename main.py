import hydra
import numpy as np
import torch
import random
import loop
from torch.utils.tensorboard import SummaryWriter

@hydra.main(config_path='config', config_name='config', version_base = '1.3.2')
def main(config):
    
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)

    #NOTE: Don't have to use tensorboard to log experiments, but should implement something else if so (Aleksis have code for this).
    writer = SummaryWriter(log_dir='raid/dlgroupmsc/test1')
    loop.loop(config)

if __name__ == '__main__':
    main()