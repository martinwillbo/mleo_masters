from datetime import datetime
import hydra
import numpy as np
import torch
import random
import loop3
import os
#import test
import stats
from torch.utils.tensorboard import SummaryWriter

@hydra.main(config_path='config', config_name='config', version_base = '1.3.2')
def main(config):
    
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)

    #NOTE: Don't have to use tensorboard to log experiments, but should implement something else if so (Aleksis has code for this).
    #name more cleverly
    current_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    hydra_log_dir = '../log_res/'+current_timestamp
    log_dir = os.path.join(hydra_log_dir, 'tensorboard')
    print(log_dir)
    writer = SummaryWriter(log_dir=log_dir)
    # writer = SummaryWriter(log_dir='.')

    mean, std = stats.stats(config)
    #loop3.loop3(config, writer, hydra_log_dir)
    #loop2.loop2(config, writer, hydra_log_dir)
    # test.eval_on_test(config, writer = None)

if __name__ == '__main__':
    main()