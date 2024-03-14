import hydra
import numpy as np
import torch
import random
import loop2
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime

@hydra.main(config_path='config', config_name='config', version_base = '1.3.2')
def main(config):
    #os.environ['TMPDIR'] = '/raid/dlgroupmsc/exjobb/tmp'

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)

    #NOTE: Don't have to use tensorboard to log experiments, but should implement something else if so (Aleksis have code for this).
    current_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    hydra_log_dir = '/raid/dlgroupmsc/logs/'+current_timestamp
    log_dir = os.path.join(hydra_log_dir, 'tensorboard')
    print(log_dir)
    writer = SummaryWriter(log_dir=log_dir)
    loop2.loop2(config, writer, hydra_log_dir)

if __name__ == '__main__':
    main()