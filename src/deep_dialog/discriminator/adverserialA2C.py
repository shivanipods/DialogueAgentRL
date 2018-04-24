from __future__ import print_function

import os
import sys
import argparse
import numpy as np
np.random.seed(0)
import tensorflow as tf
import keras
from keras.callbacks import TensorBoard
import tensorflow as tf
from keras.models import Sequential,load_model, Model
from keras.layers import Dense, Dropout, Flatten, Multiply, Activation
from keras.layers import Conv2D, MaxPooling2D, Input, Lambda
from keras.optimizers import Adam, Adamax, RMSprop
from keras.initializers import VarianceScaling
from keras import backend as K
import gym
from gym import wrappers
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from time import time
import pickle as pkl
import math


class AdverserialA2C():
	def __init__(self, env, args):
		self.env = env
		self.args = args
		self.nS = self.env.observation_space.shape[0]
		self.nA = self.env.action_space.n

		## create the expert network
		with open(self.args.model_config_path, 'r') as f:
			self.expert = keras.models.model_from_json(f.read())
		self.expert.load_weights(self.args.expert_weights_path)

		kernel = keras.initializers.VarianceScaling(scale=1.0, mode='fan_avg',
													distribution='uniform', seed=None)

		## create the discriminator
		self.discriminator = Sequential([
			Dense(20, input_shape=(self.nS + self.nA,), kernel_initializer=kernel),
			Activation('relu'),
			Dense(20, kernel_initializer=kernel),
			Activation('relu'),
			Dense(1, activation='sigmoid', kernel_initializer=kernel),
		])

		## create the actor
		with open(self.args.model_config_path, 'r') as f:
			self.actor = keras.models.model_from_json(f.read())

		## create the original critic
		self.critic = self.build_critic()

		## create the gan critic
		self.gan_critic = self.build_critic()


		self.n = args.n
		self.gamma = args.gamma

	def build_critic(self):
		model = Sequential()
		fc1 = Dense(50, input_shape=(self.nS,), activation='relu',
					kernel_initializer=VarianceScaling(mode='fan_avg',
													   distribution='normal'))
		fc2 = Dense(50, activation='relu',
					kernel_initializer=VarianceScaling(mode='fan_avg',
													   distribution='normal'))
		fc3 = Dense(1, activation='relu',
					kernel_initializer=VarianceScaling(mode='fan_avg',
													   distribution='normal'))
		model.add(fc1)
		model.add(fc2)
		model.add(fc3)
		return model

	def truncated_discounted_rewards(self, rewards):

		batch_size = len(rewards) - self.n
		truncated_rewards = np.zeros(batch_size)
		for t in range(batch_size):
			cumulative = 0
			for i in range(0, self.n):
				cumulative += math.pow(self.gamma, i) * rewards[t + i]
			truncated_rewards[t] = cumulative
		return truncated_rewards

	def get_value_reward(self, states, rewards, values):

		extended_values = values + [0] * self.n
		extended_rewards = rewards + [0] * self.n
		truncated_discounted_rewards = self.truncated_discounted_rewards(extended_rewards)
		batch_size = len(rewards)
		discounted_rewards = np.zeros_like(rewards)
		for t in reversed(range(batch_size)):
			discounted_rewards[t] = math.pow(self.gamma, self.n) * extended_values[t + self.n] + \
									truncated_discounted_rewards[t]
		return discounted_rewards

	def test_episode(self, num_test_episodes=100):
		all_rewards = []
		for i in range(num_test_episodes):
			_, _, rewards = self.generate_episode(self.env, self.args.render)
			episode_reward = np.sum(rewards)
			all_rewards.append(episode_reward * 1e2)

		average_reward = np.mean(all_rewards)
		std_reward = np.std(all_rewards)
		return average_reward, std_reward

	def generate_episode(self, env, model, render=False):

		states = []
		actions = []
		rewards = []

		current_state = env.reset()
		is_terminal = False
		while not is_terminal:
			action_distribution = self.actor.predict(np.expand_dims(current_state, 0))
			action = np.random.choice(env.action_space.n, 1,
									  p=action_distribution.squeeze(0))[0]
			next_state, reward, is_terminal, _ = env.step(action)
			states.append(current_state)
			actions.append(action)
			rewards.append(reward * 1e-2)
			current_state = next_state

		return states, actions, rewards

	def compute_baseline(self, states, isgan = False):
		values = []
		for e, state in enumerate(states):
			if isgan:
				criticval = self.gan_critic.predict(np.expand_dims(state, 0))
			else:
				criticval = self.critic.predict(np.expand_dims(state, 0))
			values.append(criticval.squeeze(0)[0])
		return values


	def train(self):
		if self.args.optimizer == "adam":
			self.actor_optimizer = Adam(lr=self.args.actor_lr)
			self.critic_optimizer = Adam(lr=self.args.critic_lr)
			self.gan_critic_optimizer = Adam(lr=self.args.critic_lr)
			self.discriminator_optimizer = Adam(lr = self.args.discriminator_lr)

		self.actor.compile(optimizer=self.actor_optimizer, loss='categorical_crossentropy')
		self.critic.compile(optimizer=self.critic_optimizer, loss='mse')
		self.gan_critic.compile(optimizer=self.gan_critic_optimizer, loss='mse')
		self.discriminator.compile(optimizer=self.critic_optimizer, loss='binary_crossentropy', metrics=['accuracy'])

		## populate training data for the discriminator
		# self.run_expert()

		total_act_loss = []
		total_crit_loss = []
		all_rewards = []
		adverserial_rewards = []


		## train
		for episode in range(self.args.num_episodes):
			## generate training example
			states, actions, rewards = self.generate_episode(self.env, self.args.render)


			''' Train normal actor critic network '''
			values = self.compute_baseline(states)
			discounted_rewards = self.get_value_reward(states, rewards, values)
			states = np.array(states)
			act_target = np.zeros((len(states), self.nA))
			act_target[np.arange(len(states)), np.array(actions)] = (np.array(discounted_rewards)
																	 - np.array(values))

			crit_target = np.array(discounted_rewards)
			actor_loss = self.actor.train_on_batch(states, act_target)
			critic_loss = self.critic.train_on_batch(states, crit_target)

			## sample from an expert episode and the current simulated episode
			## in Goodfellow's original paper, he does it k times
			expert_states, expert_actions, _ = AdverserialA2C.generate_expert_episode(self.expert, self.env)

			expert_set = []
			simulation_set = []
			for k in range(self.args.discriminator_batch_size):
				sampled_expert_index = np.random.randint(0, len(expert_states))
				one_hot_expert_action = np.zeros(self.nA)
				one_hot_expert_action[expert_actions[sampled_expert_index]] = 1
				sampled_expert_state = np.array(expert_states[sampled_expert_index])
				# sampled_expert_state = np.expand_dims(sampled_expert_state, 0)
				sampled_expert_example = np.concatenate((sampled_expert_state, one_hot_expert_action), axis=0)
				expert_set.append(sampled_expert_example)
				sampled_simulated_index = np.random.randint(0, len(states))
				one_hot_simulated_action = np.zeros(self.nA)
				one_hot_simulated_action[actions[sampled_simulated_index]] = 1
				sampled_simulated_state = states[sampled_simulated_index]
				# sampled_simulated_state = np.expand_dims(sampled_simulated_state,0)
				sampled_simulated_example = np.concatenate((sampled_simulated_state, one_hot_simulated_action), axis=0)
				simulation_set.append(sampled_simulated_example)

			expert_set = np.asarray(expert_set)
			simulation_set = np.asarray(simulation_set)
			## combined training data
			combined_training_data = np.concatenate((expert_set, simulation_set), axis=0)
			combined_prediction_values = np.concatenate((np.ones((self.args.discriminator_batch_size, 1)) ,
														 np.zeros((self.args.discriminator_batch_size, 1))), axis=0)
			p = np.random.permutation(self.args.discriminator_batch_size)
			combined_training_data = combined_training_data[p]
			combined_prediction_values = combined_prediction_values[p]

			## train discriminator
			d_loss, metric = self.discriminator.train_on_batch(combined_training_data, combined_prediction_values)


			## compute gan rewards
			## call predict on a batch of the current simulated  episodes to get the class value
			state_action_pairs = []
			for s,a in zip(states, actions):
				one_hot = np.zeros(self.nA)
				one_hot[a] = 1
				concat_s_a = np.concatenate((s, one_hot))
				state_action_pairs.append(concat_s_a)
			probability_simulation = self.discriminator.predict(np.array(state_action_pairs))
			gan_rewards = (-np.log(1-probability_simulation)).flatten().tolist()

			''' Train gan actor-critic network '''
			gan_values = self.compute_baseline(states, isgan=True)
			gan_discounted_rewards = self.get_value_reward(states, gan_rewards, gan_values)
			gan_act_target = np.zeros((len(states), self.nA))
			gan_act_target[np.arange(len(states)), np.array(actions)] = (np.array(gan_discounted_rewards)
																	 - np.array(gan_values))
			gan_critic_target = np.array(gan_discounted_rewards)
			gan_actor_loss = self.actor.train_on_batch(states, gan_act_target)
			gan_critic_loss = self.gan_critic.train_on_batch(states, gan_critic_target)


			total_act_loss.append(actor_loss + gan_actor_loss)
			total_crit_loss.append(critic_loss + gan_critic_loss)
			all_rewards.append(np.sum(rewards) * 1e2)
			adverserial_rewards.append(np.sum(gan_rewards))

			if (episode + 1) % self.args.log_every == 0:
				if self.args.verbose:
					print("Num Episodes: {0}, Train Reward: {1} +/- {2}, Act Loss: {3}, Crit Loss: {4}".
						  format(episode + 1, np.mean(all_rewards), np.std(all_rewards),
								 np.mean(total_act_loss), np.mean(total_crit_loss)))
				total_act_loss = []
				total_crit_loss = []
				all_rewards = []
				adverserial_rewards = []

			if (not self.args.trial) and (episode + 1) % self.args.eval_after == 0:
				if self.args.verbose:
					print("Saving Model Weights")
				avg_reward, std_reward = self.test_episode()
				print("Test Reward:{0} +/- {1}".format(avg_reward, std_reward))
				self.actor.save_weights(os.path.join(self.args.model_path, str(episode + 1)))

	@staticmethod
	def generate_expert_episode(model, env):
		# Generates an episode by running the given model on the given env.
		# Returns:
		# - a list of states, indexed by time step
		# - a list of actions, indexed by time step
		# - a list of rewards, indexed by time step
		states = []
		actions = []
		rewards = []

		current_state = env.reset()
		is_terminal = False
		while not is_terminal:
			action_distribution = model.predict(np.expand_dims(current_state, 0))
			action = np.random.choice(env.action_space.n, 1,
									  p=action_distribution.squeeze(0))[0]
			next_state, reward, is_terminal, _ = env.step(action)
			states.append(current_state)
			actions.append(action)
			rewards.append(reward)
			current_state = next_state

		return states, actions, rewards

	## obtain training dataset from expert policy
	def run_expert(self):
		self.discriminator_training_data = []
		# Generates an episode by running the expert policy on the given env.
		for i in range(self.args.num_expert_episodes):
			states, actions, _ = AdverserialA2C.generate_expert_episode(self.expert, self.env)
			for tuple in zip(states, actions):
				self.discriminator_training_data.append(tuple)

	def test(self):
		trained_episodes = sorted([int(ep) for ep in os.listdir(self.args.model_path)])
		performance = []
		for episode in trained_episodes:
			self.actor.load_weights(os.path.join(self.args.model_path, str(episode)))
			average_reward, std_reward = self.test_episode()
			performance.append([episode, average_reward, std_reward])
			print('*' * 80)
			print("Average reward after {0} episodes: {1} +/- {2}".format(episode,
																		  average_reward, std_reward))
			print('*' * 80)
		self.plot_performance(performance)

	def plot_performance(self, performance):

		X = [x[0] for x in performance]
		Y = [x[1] for x in performance]
		Z = [x[2] for x in performance]

		plt.figure()
		plt.errorbar(X, Y, yerr=Z, ecolor='r', capsize=2)
		plt.axhline(y=200, linestyle='--')
		plt.xlabel('Episodes')
		plt.ylabel('Average Reward')
		plt.title("Performance of A2C Algorithm (n=" + str(self.args.n) + ") on Lunar Lander")
		plt.savefig(self.args.plot_path, dpi=200)


def parse_arguments():
	parser = argparse.ArgumentParser()

	parser.add_argument('--expert-weights-path', dest='expert_weights_path', type=str,
						default='LunarLander-v2-weights.h5',
						help="Path to the expert weights file.")
	parser.add_argument('--model-config-path', dest='model_config_path', type=str,
						default='LunarLander-v2-config.json',
						help="Path to the model config file.")
	parser.add_argument('--result_path', dest='result_path', type=str,
						default='adverserial_a2c_keras',
						help="Path to the model.")
	parser.add_argument('--resume', dest='resume', type=int, default=0,
						help="Resume the training from last checkpoint")

	parser.add_argument('--num-episodes', dest='num_episodes', type=int,
						default=50000,
						help="Number of episodes to train on.")
	parser.add_argument('--eval_after', dest='eval_after', type=int,
						default=50,
						help="Number of episodes to evaluate after.")
	parser.add_argument('--log_every', dest='log_every', type=int,
						default=25,
						help="Number of episodes to log after.")
	parser.add_argument('--gamma', type=float, dest='gamma',
						default=1)

	parser.add_argument('--run', type=int, dest='run', default=1)
	parser.add_argument('--seed', type=int, dest='seed', default=0)
	parser.add_argument('--num-test-episodes', type=int, dest='num_test_episodes', default=50)
	parser.add_argument('--trial', dest='trial', action='store_true',
						help="If it is just a trial")
	parser.add_argument('--verbose', dest='verbose', action='store_true',
						help="Whether to print loss after every episode.")
	parser.add_argument('--mode', type=str, dest='mode',
						default='train',
						help="Optimizer to be Used")

	parser_group = parser.add_mutually_exclusive_group(required=False)
	parser_group.add_argument('--render', dest='render',
							  action='store_true',
							  help="Whether to render the environment.")
	parser_group.add_argument('--no-render', dest='render',
							  action='store_false',
							  help="Whether to render the environment.")
	parser.set_defaults(render=False)

	parser.add_argument('--n', dest='n', type=int,
						default=50, help="The value of N in N-step A2C.")
	parser.add_argument('--critic_lr', dest='critic_lr', type=float, default=1e-3,
						help="The learning rate.")
	parser.add_argument('--optimizer', type=str, dest='optimizer',
						default='adam', help="Optimizer to be Used")
	parser.add_argument('--actor_lr', dest='actor_lr', type=float, default=5e-4)
	parser.add_argument('--discriminator_lr', dest='discriminator_lr', type=float, default=5e-4)
	parser.add_argument('--num-expert-episodes', dest='num_expert_episodes', type=int, default=50)
	parser.add_argument('--discriminator-batch-size', dest='discriminator_batch_size', default=10)

	args = parser.parse_args()

	if not os.path.exists(args.result_path):
		os.makedirs(args.result_path)

	args.model_path = os.path.join(args.result_path,
								   'adverserial_a2c_model' + str(args.run) + '_' + str(args.n))

	if not os.path.exists(args.model_path):
		os.makedirs(args.model_path)

	args.plot_path = os.path.join(args.result_path, 'adverserial_a2c_plot' + str(args.run) +
								  '_' + str(args.n) + '.png')

	return args

def main(args):
	args = parse_arguments()

	# Create the environment.
	env = gym.make('LunarLander-v2')
	env.seed(args.seed)

	a2c = AdverserialA2C(env, args)

	if args.mode == 'train':
		a2c.train()

	elif args.mode == 'test':
		a2c.test()

if __name__ == '__main__':
	main(sys.argv)
