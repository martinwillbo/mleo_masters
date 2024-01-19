import hydra
import numpy as np
import torch
import random
import loop
import stats
from torch.utils.tensorboard import SummaryWriter

@hydra.main(config_path='config', config_name='config', version_base = '1.3.2')
def main(config):
    
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)

    #NOTE: Don't have to use tensorboard to log experiments, but should implement something else if so (Aleksis have code for this).
    #name more cleverly
    mean, std = stats.stats(config)
    print(mean)
    print(std)
    #writer = SummaryWriter(log_dir='/raid/dlgroupmsc/writer_${now:%Y-%m-%d_%H-%M-%S}')
    #loop.loop(config, writer)

if __name__ == '__main__':
    main()