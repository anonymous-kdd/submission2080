import math
import torch.nn.functional as F

def shifted_softplus(data):
    return F.softplus(data) - math.log(2)