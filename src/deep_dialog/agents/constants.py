import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T
import numpy as np

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor


def make_np_array(arr, dtype=np.float):
    return np.asarray([np.asarray(x).astype(dtype) for x in arr])

def placeholder(np_var):
    v = variable(torch.from_numpy(np_var))
    return v

def variable(v, volatile=False):
    if use_cuda:
        return Variable(v, volatile=volatile).cuda()
    return Variable(v, volatile=volatile)
