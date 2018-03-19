import random, copy, json
import cPickle as pickle
import numpy as np


from deep_dialog import dialog_config
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple
from torch.autograd import Variable

from agent_dqn_torch import AgentDQNTorch
from deep_dialog.qlearning.bayesianDQN import BayesianMLP
from deep_dialog.qlearning.utils import *


class AgentBBQN(AgentDQNTorch):
	def __init__(self, movie_dict=None, act_set=None, slot_set=None, params=None):
		super(AgentBBQN, self).__init__(movie_dict, act_set, slot_set, params)
		self.sigma_prior = params.get('sigma_prior', float(np.exp(-3)))
		## reference is being replaced
		self.dqn = BayesianMLP(self.state_dimension, self.hidden_size, self.hidden_size, self.num_actions)
		self.clone_dqn = copy.deepcopy(self.dqn)
		if torch.cuda.is_available():
			self.dqn.cuda()
			self.clone_dqn.cuda()



	def return_greedy_action(self, state_representation):
		state_var = Variable(torch.FloatTensor(state_representation).unsqueeze(0))
		if torch.cuda.is_available():
			state_var = state_var.cuda()
		qvalues = self.dqn(state_var)
		action =  qvalues.data.max(1)[1]
		return action[0]

	# def update_model_with_replay(self, batch, batch_size):
	#
	# 	batch = self.transition(*zip(*batch))
	# 	state_batch = Variable(torch.FloatTensor(batch.state))
	# 	action_batch = Variable(torch.LongTensor(batch.action))
	# 	if torch.cuda.is_available():
	# 		state_batch = state_batch.cuda()
	# 		action_batch = action_batch.cuda()
	# 	prediction = self.dqn(state_batch).gather(1, action_batch.view(batch_size, 1))
	# 	sample_log_pw, sample_log_qw = self.dqn.get_lpw_lqw()
	# 	target = Variable(torch.zeros(batch_size))
	# 	next_state_batch = Variable(torch.FloatTensor(batch.next_state), volatile=True)
	# 	nqvalues = self.clone_dqn(next_state_batch)
	# 	nqvalues = nqvalues.max(1)[0]
	# 	nqvalues.volatile = False
	# 	for i in range(batch_size):
	# 		done = batch.is_terminal[i]
	# 		target[i] = batch.reward[i] + (1 - done) * self.gamma * nqvalues[i]
	# 	if torch.cuda.is_available():
	# 		target = target.cuda()
	# 	loss = self.loss_function(prediction, target)
	# 	self.optimizer.zero_grad()
	# 	loss.backward()
	# 	self.optimizer.step()
	# 	return loss

	def update_model_with_replay(self, batch, batch_size):
		s_log_pw, s_log_qw, s_log_likelihood = 0., 0., 0.
		batch = self.transition(*zip(*batch))
		state_batch = Variable(torch.FloatTensor(batch.state))
		action_batch = Variable(torch.LongTensor(batch.action))
		if torch.cuda.is_available():
			state_batch = state_batch.cuda()
			action_batch = action_batch.cuda()
		prediction = self.dqn(state_batch).gather(1, action_batch.view(batch_size, 1))
		## this is the variational approximation (q) and posterior (p) used to compute KL divergence
		sample_log_pw, sample_log_qw = self.dqn.get_lpw_lqw()
		target = Variable(torch.zeros(batch_size))
		next_state_batch = Variable(torch.FloatTensor(batch.next_state), volatile=True)
		nqvalues = self.clone_dqn(next_state_batch, infer=True)
		nqvalues = nqvalues.max(1)[0]
		# loss = self.loss_function(prediction, target)
		for i in range(batch_size):
			done = batch.is_terminal[i]
			target[i] = batch.reward[i] + (1 - done) * self.gamma * nqvalues[i]
		likelihood_error = log_gaussian(target, prediction, self.sigma_prior).sum()
		s_log_pw = sample_log_pw.sum()
		s_log_qw = sample_log_qw.sum()
		s_log_likelihood = likelihood_error.sum()
		return s_log_pw/batch_size, s_log_qw/batch_size, s_log_likelihood/batch_size


	def return_greedy_action(self, state_representation):
		state_var = Variable(torch.FloatTensor(state_representation).unsqueeze(0))
		if torch.cuda.is_available():
			state_var = state_var.cuda()
		qvalues = self.dqn(state_var)
		action =  qvalues.data.max(1)[1]
		return action[0]


	def train(self, batch_size=1, num_batches=100):
		""" Train DQN with experience replay """
		optimizer_type = self.params.get("optimizer", "adam")
		learning_rate = self.params.get("lrate", 0.0001)
		# l2 regularization weight decay
		reg_cost = self.params.get('reg_cost', 1e-3)
		loss_function_type = self.params.get("loss_function", "mse")


		## create pytorch optimizer object
		if optimizer_type == 'rmsprop':
			self.optimizer = optim.RMSprop(self.dqn.parameters(), lr=learning_rate, weight_decay=reg_cost)
		else:
			self.optimizer = optim.Adam(self.dqn.parameters(), lr=learning_rate, weight_decay=reg_cost)

		## define loss function
		if loss_function_type == "hubert":
			self.loss_function = nn.SmoothL1Loss()
		else:
			self.loss_function = nn.MSELoss()


		for iter_batch in range(num_batches):
			self.cur_bellman_err = 0
			log_pw, log_qw, log_likelihood = 0., 0., 0.
			for iter in range(len(self.experience_replay_pool) / (batch_size)):
				batch = [random.choice(self.experience_replay_pool) for i in xrange(batch_size)]
				# batch_struct = self.dqn.singleBatch(batch, {'gamma': self.gamma}, self.clone_dqn)
				# self.cur_bellman_err += batch_struct['cost']['total_cost']
				## they are also doing L2 regularization term apart from the MSE loss, always using the target qnetwork and -1e-3 gradient clipping
				## thier learning rate is 0.001
				log_pw, log_qw, log_llh = self.update_model_with_replay(batch, batch_size)
				loss = bayes_loss_function(log_pw, log_qw, log_llh, n_batches, batch_size)
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()
				# self.cur_bellman_err += mse_loss.data.numpy().sum()


			print ("cur bellman err %.4f, experience replay pool %s" % (
			float(self.cur_bellman_err) / len(self.experience_replay_pool), len(self.experience_replay_pool)))

