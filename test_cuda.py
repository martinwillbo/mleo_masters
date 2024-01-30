import torch
import torch.nn as nn
import numpy as np

x = torch.zeros(3,3)
x[torch.randn(3,3) > 0.5] = 1

print(x)
print(torch.count_nonzero(x).item())