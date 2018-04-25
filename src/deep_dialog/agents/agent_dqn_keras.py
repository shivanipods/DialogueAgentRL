import random, copy, json
import cPickle as pickle
import numpy as np
from constants import *
import keras
from keras.initializers import VarianceScaling
from keras import regularizers
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Lambda
from keras.optimizers import Adam
from constants import *
from collections import namedtuple
import ipdb
from agent import Agent
from deep_dialog import dialog_config
from collections import deque


class Replay_Memory():
	def __init__(self, memory_size=1000, burn_in=200):
		self.memory_size = memory_size
		self.burn_in = burn_in
		self.memory = deque(maxlen=self.memory_size)

	def sample_batch(self, batch_size=32):
		return random.sample(self.memory, batch_size)

	def append(self, transition):
		self.memory.append(transition)


class AgentDQNKeras(Agent):
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
		self.learning_rate = self.params.get("lrate", 0.0001)
		self.reg_cost = self.params.get('reg_cost', 1e-3)
		## warm start:
		self.warm_start = params.get('warm_start', 0)

		self.max_turn = params['max_turn'] + 4
		self.state_dimension = 2 * self.act_cardinality + 7 * self.slot_cardinality + 3 + self.max_turn

		## TODO: keras version of DQN
		self.build_qnetwork_model()
		self.clone_dqn = keras.models.clone_model(self.dqn)
		self.clone_dqn.set_weights(self.dqn.get_weights())


		self.transition = namedtuple('transition', ('state', 'action', 'reward', 'next_state', 'is_terminal'))

		self.cur_bellman_err = 0

		## load a model if present
		if params['trained_model_path'] != None:
			self.dqn = copy.deepcopy(self.dqn.load((params['trained_model_path'])))
			self.clone_dqn = copy.deepcopy(self.dqn)
			self.predict_mode = True
			self.warm_start = 2

	def build_qnetwork_model(self):
		model = Sequential()
		fc1 = Dense(self.hidden_size, input_shape=(self.state_dimension,), activation='relu',
					kernel_initializer=VarianceScaling(mode='fan_avg', distribution='normal'),
					kernel_regularizer=regularizers.l2(self.reg_cost))
		fc2 = Dense(self.hidden_size, activation='relu',
					kernel_initializer=VarianceScaling(mode='fan_avg', distribution='normal'),
					kernel_regularizer=regularizers.l2(self.reg_cost))

		fc3 = Dense(self.num_actions, activation='linear',
					kernel_initializer=VarianceScaling(mode='fan_avg', distribution='normal'),
					kernel_regularizer=regularizers.l2(self.reg_cost))
		model.add(fc1)
		model.add(fc2)
		model.add(fc3)
		model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
		self.dqn = model


	def initialize_episode(self):
		""" Initialize a new episode. This function is called every time a new episode is run. """

		self.current_slot_id = 0
		self.phase = 0
		self.request_set = ['moviename', 'starttime', 'city', 'date', 'theater', 'numberofpeople']

	def prepare_state_representation(self, state):
		""" Create the representation for each state """

		user_action = state['user_action']
		current_slots = state['current_slots']
		kb_results_dict = state['kb_results_dict']
		agent_last = state['agent_action']

		########################################################################
		#   Create one-hot of acts to represent the current user action
		########################################################################
		user_act_rep = np.zeros((1, self.act_cardinality))
		user_act_rep[0, self.act_set[user_action['diaact']]] = 1.0

		########################################################################
		#     Create bag of inform slots representation to represent the current user action
		########################################################################
		user_inform_slots_rep = np.zeros((1, self.slot_cardinality))
		for slot in user_action['inform_slots'].keys():
			user_inform_slots_rep[0, self.slot_set[slot]] = 1.0

		########################################################################
		#   Create bag of request slots representation to represent the current user action
		########################################################################
		user_request_slots_rep = np.zeros((1, self.slot_cardinality))
		for slot in user_action['request_slots'].keys():
			user_request_slots_rep[0, self.slot_set[slot]] = 1.0

		########################################################################
		#   Creat bag of filled_in slots based on the current_slots
		########################################################################
		current_slots_rep = np.zeros((1, self.slot_cardinality))
		for slot in current_slots['inform_slots']:
			current_slots_rep[0, self.slot_set[slot]] = 1.0

		########################################################################
		#   Encode last agent act
		########################################################################
		agent_act_rep = np.zeros((1, self.act_cardinality))
		if agent_last:
			agent_act_rep[0, self.act_set[agent_last['diaact']]] = 1.0

		########################################################################
		#   Encode last agent inform slots
		########################################################################
		agent_inform_slots_rep = np.zeros((1, self.slot_cardinality))
		if agent_last:
			for slot in agent_last['inform_slots'].keys():
				agent_inform_slots_rep[0, self.slot_set[slot]] = 1.0

		########################################################################
		#   Encode last agent request slots
		########################################################################
		agent_request_slots_rep = np.zeros((1, self.slot_cardinality))
		if agent_last:
			for slot in agent_last['request_slots'].keys():
				agent_request_slots_rep[0, self.slot_set[slot]] = 1.0

		turn_rep = np.zeros((1, 1)) + state['turn'] / 10.

		########################################################################
		#  One-hot representation of the turn count?
		########################################################################
		turn_onehot_rep = np.zeros((1, self.max_turn))
		turn_onehot_rep[0, state['turn']] = 1.0

		########################################################################
		#   Representation of KB results (scaled counts)
		########################################################################
		kb_count_rep = np.zeros((1, self.slot_cardinality + 1)) + kb_results_dict['matching_all_constraints'] / 100.
		for slot in kb_results_dict:
			if slot in self.slot_set:
				kb_count_rep[0, self.slot_set[slot]] = kb_results_dict[slot] / 100.

		########################################################################
		#   Representation of KB results (binary)
		########################################################################
		kb_binary_rep = np.zeros((1, self.slot_cardinality + 1)) + np.sum(
			kb_results_dict['matching_all_constraints'] > 0.)
		for slot in kb_results_dict:
			if slot in self.slot_set:
				kb_binary_rep[0, self.slot_set[slot]] = np.sum(kb_results_dict[slot] > 0.)

		self.final_representation = np.hstack(
			[user_act_rep, user_inform_slots_rep, user_request_slots_rep, agent_act_rep, agent_inform_slots_rep,
			 agent_request_slots_rep, current_slots_rep, turn_rep, turn_onehot_rep, kb_binary_rep, kb_count_rep])
		self.final_representation = np.squeeze(self.final_representation)
		return self.final_representation

	def register_experience_replay_tuple(self, s_t, a_t, reward, s_tplus1, episode_over):
		""" Register feedback from the environment, to be stored as future training data """

		state_t_rep = self.prepare_state_representation(s_t)
		# from where does it get this action assigned to self
		action_t = self.action
		reward_t = reward
		state_tplus1_rep = self.prepare_state_representation(s_tplus1)
		training_example = (state_t_rep, action_t, reward_t, state_tplus1_rep, episode_over)

		if self.predict_mode == False:  # Training Mode
			if self.warm_start == 1:
				self.experience_replay_pool.append(training_example)
		else:
			# Prediction Mode
			self.experience_replay_pool.append(training_example)

	def state_to_action(self, state):
		""" DQN: Input state, output action """
		## Dialogue manager calls this to fill the experience buffer ##
		self.representation = self.prepare_state_representation(state)
		self.action = self.run_policy(self.representation)
		act_slot_response = copy.deepcopy(self.feasible_actions[self.action])
		return {'act_slot_response': act_slot_response, 'act_slot_value_response': None}, None

	def rule_policy(self):
		""" Rule Policy """

		if self.current_slot_id < len(self.request_set):
			slot = self.request_set[self.current_slot_id]
			self.current_slot_id += 1

			act_slot_response = {}
			act_slot_response['diaact'] = "request"
			act_slot_response['inform_slots'] = {}
			act_slot_response['request_slots'] = {slot: "UNK"}
		elif self.phase == 0:
			act_slot_response = {'diaact': "inform", 'inform_slots': {'taskcomplete': "PLACEHOLDER"},
								 'request_slots': {}}
			self.phase += 1
		elif self.phase == 1:
			act_slot_response = {'diaact': "thanks", 'inform_slots': {}, 'request_slots': {}}

		return self.action_index(act_slot_response)

	def action_index(self, act_slot_response):
		""" Return the index of action """

		for (i, action) in enumerate(self.feasible_actions):
			if act_slot_response == action:
				return i
		print act_slot_response
		raise Exception("action index not found")
		return None

	def return_greedy_action(self, state_representation):
		state_tensor= np.expand_dims(np.asarray(state_representation), axis=0)
		qvalues = self.dqn.predict(state_tensor)
		action = np.argmax(qvalues)
		return action

	def run_policy(self, representation):
		""" epsilon-greedy policy """

		if random.random() < self.epsilon:
			return random.randint(0, self.num_actions - 1)
		else:
			if self.warm_start == 1:
				## if in training mode(not prediction) fill until you cant anymore
				if len(self.experience_replay_pool) > self.experience_replay_pool_size:
					self.warm_start = 2
				return self.rule_policy()
			else:
				# return self.dqn.predict(representation, {}, predict_model=True)
				return self.return_greedy_action(representation)

	def update_model_with_replay(self, batch, batch_size):
		batch = self.transition(*zip(*batch))
		state_batch = np.asarray(batch.state)
		action_batch = np.asarray(batch.action)
		targets = self.dqn.predict(state_batch)
		# prediction = prediction[np.arange(batch_size), action_batch]
		next_state_batch = np.asarray(batch.next_state)
		nqvalues = self.clone_dqn.predict(next_state_batch)
		nqvalues = nqvalues.max(1)[0]
		mask = np.asarray(batch.is_terminal)
		reward = np.asarray(batch.reward)
		temp = self.gamma * nqvalues
		# ipdb.set_trace()
		target = reward + temp*(1 - mask)
		## only replace the relevant actions
		targets[np.arange(batch_size), action_batch] = target
		loss = self.dqn.train_on_batch(state_batch, targets)
		return loss

	def train(self, batch_size=1, num_batches=100):
		for iter_batch in range(num_batches):
			self.cur_bellman_err = 0
			for iter in range(len(self.experience_replay_pool) / (batch_size)):
				batch = [random.choice(self.experience_replay_pool) for i in xrange(batch_size)]
				## they are also doing L2 regularization term apart from the MSE loss, always using the target qnetwork and -1e-3 gradient clipping
				## thier learning rate is 0.001
				mse_loss = self.update_model_with_replay(batch, batch_size)
				self.cur_bellman_err += mse_loss

			print ("cur bellman err %.4f, experience replay pool %s" % (
				float(self.cur_bellman_err) / len(self.experience_replay_pool), len(self.experience_replay_pool)))

	################################################################################
	#    Debug Functions
	################################################################################
	def save_experience_replay_to_file(self, path):
		""" Save the experience replay pool to a file """

		try:
			pickle.dump(self.experience_replay_pool, open(path, "wb"))
			print 'saved model in %s' % (path,)
		except Exception, e:
			print 'Error: Writing model fails: %s' % (path,)
			print e

	def load_experience_replay_from_file(self, path):
		""" Load the experience replay pool from a file"""

		self.experience_replay_pool = pickle.load(open(path, 'rb'))