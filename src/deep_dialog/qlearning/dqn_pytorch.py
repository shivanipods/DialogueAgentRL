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

class QNetwork(nn.Module):
	def __init__(self):
		super(QNetwork, self).__init__()
		pass

	def save_model(self, path):
		pass

	def load_model(self, path):
		pass

class LinearQNetwork(QNetwork):

	def __init__(self, input_size, output_size):
		super(LinearQNetwork, self).__init__()
		self.linear_layer = nn.Linear(input_size, output_size)
		self.relu = nn.ReLU()


	def forward(self, input):
		output1 = self.linear_layer(input)
		output = self.relu(output1)
		return output

class MultiLayerQNetwork(nn.Module):
	def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
		super(MultiLayerQNetwork, self).__init__()
		self.mlp_layer = nn.Sequential(torch.nn.Linear(input_size, hidden_size1),
									   torch.nn.ReLU(),
                                                                           torch.nn.Dropout(p=0.2),
									   torch.nn.Linear(hidden_size1, hidden_size2),
									   torch.nn.ReLU(),
                                                                           torch.nn.Dropout(p=0.2),
									   torch.nn.Linear(hidden_size2, output_size))

	def forward(self, input):
		output = self.mlp_layer(input)

		return output

class Replay_Memory():

	def __init__(self, memory_size=50000, burn_in=10000):

		# The memory essentially stores transitions recorder from the agent
		# taking actions in the environment.

		# Burn in episodes define the number of episodes that are written into the memory from the 
		# randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced. 
		# A simple (if not the most efficient) was to implement the memory is as a list of transitions.
		pass

	def sample_batch(self, batch_size=32):
		# This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples. 
		# You will feed this to your model to train.
		pass

	def append(self, transition):
		# Appends transition to the memory. 	
		pass

class DQN_Agent(object):
	def __init__(self, environment, render=False):
		self.env = environment
		self.nA = environment.action_space.n

	def get_epsilon_greedy_policy(self, q_network, epsilon):
		## q_values are Vector
		def get_epsilon_greedy_policy(state):
			action_distribution = np.ones(self.nA, dtype=float) * epsilon /self.nA
			q_values = q_network(state).data.numpy()
			optimum_action = np.argmax(q_values)
			action_distribution[optimum_action] += (1.0 - epsilon)
			return action_distribution
		return get_epsilon_greedy_policy

	def get_greedy_policy(self, q_network):
		def greedy_policy(state):
			action_distribution = np.zeros(self.nA, dtype=float)
			q_values = q_network(state).data.numpy()
			action_distribution[np.argmax(q_values)] = 1
			return action_distribution
		return greedy_policy


	def train(self):
		# In this function, we will train our network. 
		# If training without experience replay_memory, then you will interact with the environment 
		# in this function, while also updating your network parameters. 

		# If you are using a replay memory, you should interact with environment here, and store these 
		# transitions to memory, while also updating your model.
		pass

	def test(self, model_file=None):
		# Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
		# Here you need to interact with the environment, irrespective of whether you are using a memory.
		pass

	def burn_in_memory(self):
		# Initialize your replay memory with a burn_in number of episodes / transitions
		pass

class Linear_DQN_Agent(DQN_Agent):
	def __init__(self, environment, render=False):
		super(Linear_DQN_Agent, self).__init__(environment, render)
		self.nS = self.env.observation_space.shape[0]
		self.q_network = LinearQNetwork(self.nS, self.nA)
		self.greedy_policy = np.ones(self.nA, dtype=float)/self.nA
		self.target_linear_layer = nn.Linear(self.nS, self.nA, bias=False)
		## initialize both networks with the same weight
		## self.target_linear_layer = deepcopy(self.q_network)
		self.target_linear_layer.load_state_dict(self.q_network.linear_layer.state_dict())
		self.target_linear_layer.weight.requires_grad = False


	def train(self, args):

		## optimizer
		optimiser = torch.optim.Adam(self.q_network.parameters(), lr=args.learning_rate)
		## loss criterion
		criterion = nn.MSELoss()
		update_count = 0
		epsilon = args.epsilon_start

		for iter in range(1, args.num_iters + 1):
			# generate a policy
			self.e_policy = self.get_epsilon_greedy_policy(self.q_network, epsilon)
			# observe some current state
			state = self.env.reset()
			state_variable = Variable(FloatTensor(state))

			for timestep in itertools.count():

				action_distribution = self.e_policy(state_variable)
				current_action = np.random.choice(a=self.nA, p=action_distribution)

				next_state, reward, terminated, _ = self.env.step(current_action)
				update_count += 1
				next_state_variable = Variable(FloatTensor(next_state))

				q_values_current = self.q_network(state_variable)
				q_values_next = self.target_linear_layer(next_state_variable)
				td_target = reward + args.discount_factor*torch.max(q_values_next)

				#MSE loss
				loss = criterion(q_values_current[current_action], td_target)
				loss.backward()
				optimiser.step()


				if (update_count+1)%args.epsilon_decay_at and epsilon > args.epsilon_end:
					epsilon = epsilon*args.epsilon_decay_rate

				# parser.add_argument('--record_video', dest='record_video', type=int)

				if terminated:
					print("Q-function Loss at the end of episode: {0}".format(loss.data[0]))
					print("Length of episode: {0}".format(timestep))
					break

				#state = next_state
				state_variable = next_state_variable

			'''
			if iter%args.lazy_update == 0:
				# iterpolate target linear weights with linear weights
				# self.target_linear_layer.load_state_dict(args.interpolation_factor*self.target_linear_layer.state_dict() + (1 - args.interpolation_factor)*self.q_network.linear_layer.state_dict())
				interpolated_dict = self.target_linear_layer.state_dict()
				interpolated_dict["weight"] = args.interpolation_factor*interpolated_dict["weight"] + (1 - args.interpolation_factor)*self.q_network.linear.state_dict["weight"]
				# for k,v in self.target_linear_layer.state_dict().items():
				# 	interpolated_dict[k] = args.interpolation_factor*v + (1 - args.interpolation_factor)*self.q_network.linear.state_dict()[k]
				self.target_linear_layer.load_state_dict(interpolated_dict)
				# self.target_linear_layer.weight = deepcopy(args.interpolation_factor*self.target_linear_layer.weight + (1 - args.interpolation_factor)*self.q_network.linear_layer.weight)
				self.target_linear_layer.weight.requires_grad = False
				# set current weights to target weights
				# self.q_network.linear_layer = deepcopy(self.q_network.target_linear_layer)
				self.q_network.linear_layer.load_state_dict(self.target_linear_layer.state_dict())
				# self.q_network.linear_layer.weight.requires_grad = True
			'''

			if iter == args.num_iters%4 or iter == 1:
				print("recording video")

			if iter%args.lazy_update == 0:
				self.target_linear_layer.state_dict()['weight'] = args.interpolation_factor * self.target_linear_layer.weight + args.interpolation_factor * self.q_network.linear_layer.weight
				self.target_linear_layer.weight.requires_grad = False
				self.q_network.linear_layer.load_state_dict(self.target_linear_layer.state_dict())

class MLP_DQN_Agent(DQN_Agent):
	def __init__(self, environment, render=False):
		super(MLP_DQN_Agent, self).__init__(environment, render)
		self.nS = self.env.observation_space.shape[0]
		self.q_network = MultiLayerQNetwork(self.nS, 16, 16, self.nA)
		self.greedy_policy = np.ones(self.nA, dtype=float) / self.nA
		# self.target = MultiLayerQNetwork(self.nS, 16, 16, self.nA)
		## initialize both networks with the same weight
		## self.target_linear_layer = deepcopy(self.q_network)
		# self.target.load_state_dict(self.q_network.state_dict())
		# self.target.mlp_layer.weight.requires_grad = False

	def train(self, args):

		## optimizer
		optimiser = torch.optim.Adam(self.q_network.parameters(), lr=args.learning_rate)
		## loss criterion
		criterion = nn.MSELoss()
		update_count = 0
		epsilon = args.epsilon_start

		for iter in range(1, args.num_iters + 1):
			# generate a policy
			self.e_policy = self.get_epsilon_greedy_policy(self.q_network, epsilon)
			# observe some current state
			state = self.env.reset()
			state_variable = Variable(FloatTensor(state))

			for timestep in itertools.count():

				action_distribution = self.e_policy(state_variable)
				current_action = np.random.choice(a=self.nA, p=action_distribution)

				next_state, reward, terminated, _ = self.env.step(current_action)
				update_count += 1


				q_values_current = self.q_network(state_variable)
				# non_final_mask = ByteTensor([terminated])
				self.q_network.train(False)
				next_state_variable = Variable(FloatTensor(next_state))
				q_values_next = Variable(torch.zeros(self.nA).type(FloatTensor))
				q_values_next = self.q_network(next_state_variable)
				# q_values_next
				td_target = reward + args.discount_factor * torch.max(q_values_next)
				self.q_network.train(True)
				# MSE loss
				loss = criterion(q_values_current[current_action], td_target)
				loss.backward()
				optimiser.step()

				if (update_count + 1) % args.epsilon_decay_at and epsilon > args.epsilon_end:
					epsilon = epsilon * args.epsilon_decay_rate

				# parser.add_argument('--record_video', dest='record_video', type=int)

				if terminated:
					print("Q-function Loss at the end of episode: {0}".format(loss.data[0]))
					print("Length of episode: {0}".format(timestep))
					break

				# state = next_state
				state_variable = next_state_variable

			'''
			if iter%args.lazy_update == 0:
				# iterpolate target linear weights with linear weights
				# self.target_linear_layer.load_state_dict(args.interpolation_factor*self.target_linear_layer.state_dict() + (1 - args.interpolation_factor)*self.q_network.linear_layer.state_dict())
				interpolated_dict = self.target_linear_layer.state_dict()
				interpolated_dict["weight"] = args.interpolation_factor*interpolated_dict["weight"] + (1 - args.interpolation_factor)*self.q_network.linear.state_dict["weight"]
				# for k,v in self.target_linear_layer.state_dict().items():
				# 	interpolated_dict[k] = args.interpolation_factor*v + (1 - args.interpolation_factor)*self.q_network.linear.state_dict()[k]
				self.target_linear_layer.load_state_dict(interpolated_dict)
				# self.target_linear_layer.weight = deepcopy(args.interpolation_factor*self.target_linear_layer.weight + (1 - args.interpolation_factor)*self.q_network.linear_layer.weight)
				self.target_linear_layer.weight.requires_grad = False
				# set current weights to target weights
				# self.q_network.linear_layer = deepcopy(self.q_network.target_linear_layer)
				self.q_network.linear_layer.load_state_dict(self.target_linear_layer.state_dict())
				# self.q_network.linear_layer.weight.requires_grad = True
			'''

			if iter == args.num_iters % 4 or iter == 1:
				print("recording video")

			# if iter % args.lazy_update == 0:
			# 	self.target.mlp_layer.state_dict()[
			# 		'weight'] = args.interpolation_factor * self.target.mlp_layer.weight + args.interpolation_factor * self.q_network.mlp_layer.weight
			# 	# self.target.weight.requires_grad = False
			# 	self.q_network.load_state_dict(self.target.state_dict())

	def test(self, args, model_file=None):
		#self.e_policy is the online policy, dont use for evaluation??
		epsilon_greedy_policy = self.get_epsilon_greedy_policy(self.q_network, args.epsilon_test)
		average_reward = 0
		for iter in range(100):
			reward_per_episode = 0
			start_state = self.env.reset()
			for timestep in itertools.count():
				action_distribution = self.g_policy(start_state)
				# picktop
				action= np.random.choice(a=self.nA, p=action_distribution)
				next_state, reward, terminated, _ = self.env.step(action)
				reward_per_episode += reward
			average_reward += reward_per_episode
		average_reward = float(average_reward)/100
		print("Average reward per episode: {0}".format(average_reward))
		## for 100 terminating episodes get average reward and print

def parse_arguments():
	parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
	parser.add_argument('--experience', dest='experience', default=False, action="store_true")
	parser.add_argument('--env',dest='env',type=str, default="CartPole-v0")
	parser.add_argument('--render',dest='render',type=int,default=0)
	parser.add_argument('--train',dest='train',type=int,default=1)
	parser.add_argument('--model',dest='model_file',type=str)
	parser.add_argument('--discount_factor', dest='discount_factor', type=float,default=0.99)
	parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=0.00005)
	parser.add_argument('--epsilon_test', dest='epsilon_test', type=float, default=0.05)
	parser.add_argument('--epsilon_start', dest='epsilon_start', type=float, default=0.5)
	parser.add_argument('--epsilon_end', dest='epsilon_end', type=float, default=0.05)
	parser.add_argument('--epsilon_decay_at', dest='epsilon_decay_at', type=float, default=10000)
	parser.add_argument('--epsilon_decay_rate', dest='epsilon_decay_rate', type=float, default=0.1)
	parser.add_argument('--num_iters', dest='num_iters', type=int, default=10000)
	parser.add_argument('--rb_size', dest='rb_size', type=int, default=50000)
	parser.add_argument('--batch_size', dest='batch_size', type=int, default=32)
	parser.add_argument('--interpolation_factor', dest='interpolation_factor', type=float, default=0.9)
	parser.add_argument('--lazy_update', dest='lazy_update', type=float, default=10)
	return parser.parse_args()

def main(args):

	args = parse_arguments()
	environment_name = args.env

	environment = gym.make(environment_name)
	environment.reset()

	dqn_agent = MLP_DQN_Agent(environment)
	dqn_agent.train(args)
	# dqn_agent.test(args)

if __name__ == '__main__':
	main(sys.argv)

