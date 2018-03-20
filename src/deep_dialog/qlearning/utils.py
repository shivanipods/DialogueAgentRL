'''
Created on Jun 18, 2016

@author: xiul
'''

import numpy as np
import math
import torch
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()
Tensor = torch.cuda.Tensor if use_cuda else torch.Tensor
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor


def initWeight(n,d):
    scale_factor = math.sqrt(float(6)/(n + d))
    #scale_factor = 0.1
    return (np.random.rand(n,d)*2-1)*scale_factor

""" for all k in d0, d0 += d1 . d's are dictionaries of key -> numpy array """
def mergeDicts(d0, d1):
    for k in d1:
        if k in d0:
            d0[k] += d1[k]
        else:
            d0[k] = d1[k]

def variable(v, volatile=False):
    if use_cuda:
        return Variable(v, volatile=volatile).cuda()
    return Variable(v, volatile=volatile)

def log_gaussian(x, mu, sigma):
    return float(-0.5 * np.log(2 * np.pi) - np.log(np.abs(sigma))) - (x - mu)**2 / (2 * sigma**2)

def log_gaussian_logsigma(x, mu, logsigma):
    return float(-0.5 * np.log(2 * np.pi)) - logsigma - (x - mu)**2 / (2 * torch.exp(logsigma)**2)

def bayes_loss_function(l_pw, l_qw, l_likelihood, n_batches, batch_size):
    return (1./n_batches) * (l_qw - l_pw) - l_likelihood #/ float(batch_size)