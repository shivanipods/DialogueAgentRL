import random, copy
import random, copy, json
import cPickle as pickle
import numpy as np
from agent_dqn_torch import AgentDQNTorch
from agent import Agent
from deep_dialog.qlearning.bayesianDQN import BayesianMLP
from deep_dialog.qlearning.utils import *
from deep_dialog import dialog_config
from collections import namedtuple
from constants import *

class AgentBBQN(AgentDQNTorch):
	def __init__(self, movie_dict=None, act_set=None, slot_set=None, params=None):

		## parameters associated with dialogue action and slot filling
		self.movie_dict = movie_dict
		self.act_set = act_set
		self.slot_set = slot_set
		self.act_cardinality = len(act_set.keys())
		self.slot_cardinality = len(slot_set.keys())

		self.feasible_actions = dialog_config.feasible_actions
		self.num_actions = len(self.feasible_actions)

		# rl specific parameters
		# epsilon:
		self.params = params
		self.epsilon = params['epsilon']
		#
		self.agent_run_mode = params['agent_run_mode']
		#
		self.agent_act_level = params['agent_act_level']
		# experience replay
		self.experience_replay_pool_size = params.get('experience_replay_pool_size', 1000)
		self.experience_replay_pool = []  # Replay_Memory(self.experience_replay_pool_size)
		self.hidden_size = params.get('dqn_hidden_size', 60)
		# gamma : discount factor
		self.gamma = params.get('gamma', 0.9)
		self.predict_mode = params.get('predict_mode', False)
		## warm start:
		self.warm_start = params.get('warm_start', 0)

		self.max_turn = params['max_turn'] + 4
		self.state_dimension = 2 * self.act_cardinality + 7 * self.slot_cardinality + 3 + self.max_turn
		self.transition = namedtuple('transition', ('state', 'action', 'reward', 'next_state', 'is_terminal'))
		self.cur_bellman_err = 0

		## BBQN specific parameters
		self.sigma_prior = params.get('sigma_prior', float(np.exp(-3)))
		self.target_sample_type = params.get('target_sample_type', 'map')


		self.dqn = BayesianMLP(self.state_dimension, self.hidden_size, self.hidden_size, self.num_actions, self.sigma_prior)
		self.clone_dqn = copy.deepcopy(self.dqn)
		if torch.cuda.is_available():
			self.dqn.cuda()
			self.clone_dqn.cuda()



	def return_greedy_action(self, state_representation):
		state_var = variable(torch.FloatTensor(state_representation).unsqueeze(0))
		_,_,qvalues = self.dqn(state_var, infer=True)
		action =  qvalues.data.max(1)[1]
		return action[0]

	def update_model_with_replay(self, batch, batch_size):
		num_samples = 3
		batch = self.transition(*zip(*batch))
		state_batch = variable(torch.FloatTensor(batch.state))
		action_batch = variable(torch.LongTensor(batch.action))
		prediction_list = []
		sample_log_pw, sample_log_qw = 0., 0.
		for i in range(num_samples):
			log_pw, log_qw, qvalues = self.dqn(state_batch)
			prediction_list.append(qvalues.gather(1, action_batch.view(batch_size, 1)))
			sample_log_qw += log_qw
			sample_log_pw += log_pw
		# prediction = qvalues.gather(1, action_batch.view(batch_size, 1))
		## this is the variational approximation (q) and posterior (p) used to compute KL divergence
		target = variable(torch.zeros(batch_size))
		next_state_batch = variable(torch.FloatTensor(batch.next_state), volatile=True)
		if self.target_sample_type == 'map':
			_,_,nqvalues = self.clone_dqn(next_state_batch, infer=True)
		else:
			_, _, nqvalues = self.clone_dqn(next_state_batch)
		nqvalues = nqvalues.max(1)[0]
		nqvalues.volatile = False
		mask = torch.FloatTensor(batch.is_terminal)
		reward = torch.FloatTensor(batch.reward)
		temp = self.gamma * nqvalues
		target = variable(reward) + temp.mul(variable(1 - mask))
		likelihood_error = 0
		for i in range(num_samples):
			likelihood_error += log_gaussian(target, prediction_list[i], self.sigma_prior).sum()
		# for i in range(num_samples):
		# 	likelihood_error += (-self.loss_function(prediction_list[i], target))

		return sample_log_pw/num_samples, sample_log_qw/num_samples, likelihood_error/num_samples



	def train(self, batch_size=1, num_epochs=100):
		""" Train DQN with experience replay """
		optimizer_type = self.params.get("optimizer", "adam")
		learning_rate = self.params.get("lrate", 0.0001)
		# l2 regularization weight decay
		reg_cost = self.params.get('reg_cost', 1e-3)
		loss_function_type = self.params.get("loss_function", "mse")


		## create pytorch optimizer object
		if optimizer_type == 'rmsprop':
			self.optimizer = optim.RMSprop(self.dqn.parameters(), lr=learning_rate) #, weight_decay=reg_cost)
		else:
			self.optimizer = optim.Adam(self.dqn.parameters(), lr=learning_rate) #, weight_decay=reg_cost)

		## define loss function
		if loss_function_type == "hubert":
			self.loss_function = nn.SmoothL1Loss()
		else:
			self.loss_function = nn.MSELoss()


		for iter_batch in range(num_epochs):
			self.cur_bellman_err = 0
			self.kl_divergence = 0
			self.likelihood = 0
			log_pw, log_qw, log_likelihood = 0., 0., 0.
			n_batches = len(self.experience_replay_pool) / (batch_size)
			for iter in range(n_batches):
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
				self.cur_bellman_err += loss.data.cpu().numpy().sum()
				self.kl_divergence += (1./n_batches) * (log_qw - log_pw) * (1./batch_size)
				self.likelihood += (log_llh/batch_size) #must be positive

			print ("cur bellman err %.4f, experience replay pool %s" % (
			float(self.cur_bellman_err) / len(self.experience_replay_pool), len(self.experience_replay_pool)))
			print("curr kl divergence %.4f, curr likelihood(mse) %.4f" % (float(self.kl_divergence)/len(self.experience_replay_pool), float(self.likelihood)/len(self.experience_replay_pool)))

