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
from keras import backend as K
import gym
from gym import wrappers
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from time import time
import pickle as pkl
import math


class A2C():
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
		'''
                with open(self.args.model_config_path, 'r') as f:
			self.critic = keras.models.model_from_json(f.read())
		self.critic.pop()
		self.critic.add(Dense(1, activation='linear', kernel_initializer=kernel))
                '''

	        model = Sequential()
		fc1 = Dense(50, input_shape=(self.nS,),activation='relu',
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
		#model.compile(loss='mse',optimizer=Adam(lr=critic_lr))
		self.critic = model 	
                ## create the gan critic
		with open(self.args.model_config_path, 'r') as f:
			self.gan_critic = keras.models.model_from_json(f.read())
		self.gan_critic.pop()
		self.gan_critic.add(Dense(1, activation='linear', kernel_initializer=kernel))


		self.n = args.n
		self.gamma = args.gamma

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

	def compute_discriminator_reward(self, states, actions):
		return

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

			total_act_loss.append(actor_loss)
			total_crit_loss.append(critic_loss)
			all_rewards.append(np.sum(rewards) * 1e2)

			if (episode + 1) % self.args.log_every == 0:
				if self.args.verbose:
					print("Num Episodes: {0}, Train Reward: {1} +/- {2}, Act Loss: {3}, Crit Loss: {4}".
						  format(episode + 1, np.mean(all_rewards), np.std(all_rewards),
								 np.mean(total_act_loss), np.mean(total_crit_loss)))
				total_act_loss = []
				total_crit_loss = []
				all_rewards = []

			if (not self.args.trial) and (episode + 1) % self.args.eval_after == 0:
				if self.args.verbose:
					print("Saving Model Weights")
				avg_reward, std_reward = self.test_episode()
				print("Test Reward:{0} +/- {1}".format(avg_reward, std_reward))
				self.actor.save_weights(os.path.join(self.args.model_path, str(episode + 1)))

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
						default='a2c_keras',
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

	args = parser.parse_args()

	if not os.path.exists(args.result_path):
		os.makedirs(args.result_path)

	args.model_path = os.path.join(args.result_path,
								   'a2c_model' + str(args.run) + '_' + str(args.n))

	if not os.path.exists(args.model_path):
		os.makedirs(args.model_path)

	args.plot_path = os.path.join(args.result_path, 'a2c_plot' + str(args.run) +
								  '_' + str(args.n) + '.png')

	return args

def main(args):
	args = parse_arguments()

	# Create the environment.
	env = gym.make('LunarLander-v2')
	env.seed(args.seed)

	a2c = A2C(env, args)

	if args.mode == 'train':
		a2c.train()

	elif args.mode == 'test':
		a2c.test()

if __name__ == '__main__':
	main(sys.argv)
