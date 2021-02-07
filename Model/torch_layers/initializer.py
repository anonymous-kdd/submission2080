import torch
import torch.nn as nn

def GlorotOrthogonal(tensor, scale=2.):
    nn.init.orthogonal_(tensor)
    tensor.mul_(torch.sqrt(scale / ((tensor.size(0) + tensor.size(1)) * torch.var(tensor, unbiased=False))))