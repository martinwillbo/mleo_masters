import torch

if torch.cuda.is_available():
    print('Available')
else:
    print('Not available')
