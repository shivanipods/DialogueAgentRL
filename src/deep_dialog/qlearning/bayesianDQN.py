#!/usr/bin/env python
import numpy as np, gym, sys, copy, argparse
import torch
from torch import autograd
from torch import nn
from torch.autograd import Variable
from torch import FloatTensor, IntTensor, ByteTensor
from torch.nn import functional
import itertools
from copy import deepcopy
# from agents.constants import *
import numpy as np
from utils import *

class BayesianLinear(nn.Module):
	def __init__(self, input_size, output_size, sigma_prior):
		super(BayesianLinear, self).__init__()
		self.input_size = input_size
		self.output_size = output_size
		self.sigma_prior = sigma_prior

		self.W_mu = nn.Parameter(torch.Tensor(input_size, output_size).normal_(0, 0.01))
		self.W_logsigma = nn.Parameter(torch.Tensor(input_size, output_size).normal_(0, 0.01))
		self.b_mu = nn.Parameter(torch.Tensor(output_size).uniform_(-0.01, 0.01))
		self.b_logsigma = nn.Parameter(torch.Tensor(output_size).uniform_(-0.01, 0.01))

		# self.lpw = variable(Tensor([0]))
		# self.lqw = variable(Tensor([0]))

	def forward(self, input, infer=False):
		batch_size = input.size()[0]

		if infer:
			output = torch.mm(input, self.W_mu) + self.b_mu.expand(batch_size, self.output_size)
			return 0,0,output

		# non-parametrized randomization
		epsilon_W = variable(torch.Tensor(self.input_size, self.output_size).normal_(0, self.sigma_prior))
		epsilon_b = variable(torch.Tensor(self.output_size).normal_(0, self.sigma_prior))

		W = self.W_mu + torch.log(1 + torch.exp(self.W_logsigma)) * epsilon_W
		b = self.b_mu + torch.log(1 + torch.exp(self.b_logsigma)) * epsilon_b
		output = torch.mm(input, W) + b.expand(batch_size, self.output_size)

		lpw = log_gaussian(W, 0, self.sigma_prior).sum() + log_gaussian(b, 0, self.sigma_prior).sum()
		lqw = (log_gaussian_logsigma(W, self.W_mu, self.W_logsigma).sum() +
					log_gaussian_logsigma(b, self.b_mu, self.b_logsigma).sum())

		return lpw, lqw, output

class BayesianMLP(nn.Module):
	def __init__(self, input_size, hidden_size1, hidden_size2, output_size, sigma_prior):
		super(BayesianMLP, self).__init__()
		self.layer1 = BayesianLinear(input_size, hidden_size1, sigma_prior)
		self.layer2 = BayesianLinear(hidden_size1, hidden_size2, sigma_prior)
		self.layer3 = BayesianLinear(hidden_size2, output_size, sigma_prior)

	def forward(self, input, infer=False):
		lpw1, lqw1, output1 = self.layer1(input, infer)
		hidden1 = functional.relu(output1)
		lpw2, lqw2, output2 = self.layer2(hidden1, infer)
		hidden2 = functional.relu(output2)
		lpw3, lqw3, output = self.layer3(hidden2, infer)
		lpw = lpw1 + lpw2 + lpw2
		lqw = lqw1 + lqw2 + lqw3
		return lpw, lqw, output

	# def get_lpw_lqw(self):
	# 	lpw = self.layer1.lpw + self.layer2.lpw + self.layer3.lpw
	# 	lqw = self.layer1.lqw + self.layer2.lqw + self.layer3.lqw
	# 	return lpw, lqw

