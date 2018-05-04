from deep_dialog import dialog_config
from collections import deque
from agent import Agent
## so you can remove extreeous agent information in the dqn-pytorch file
import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from datetime import datetime
import json, copy
import logging
import keras
from keras.initializers import VarianceScaling
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Lambda
from keras.optimizers import Adam
from keras import regularizers
import ipdb
from constants import *
import random
import gym

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True,
                                        allow_soft_placement=True))
keras.backend.set_session(sess)

def one_hot(action, categories=4):
    x = np.zeros(categories)
    x[action] = 1
    return x


class AgentSharedA2C(Agent):
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
        self.reg_cost = self.params.get('reg_cost', 1e-3)
        self.agent_act_level = params['agent_act_level']

        ## epislon policy
        self.eps_fixed = params.get("eps_fixed", False)
        self.eps_strat = params.get("eps_strat", "linear_decay")
        self.eps_start = params.get('eps_start', 0.3)
        self.eps_end = params.get('eps_end', 0)
        self.eps_decay = params.get('eps_decay', 1e3)


        self.hidden_size = params.get('a2c_hidden_size', 50)
        # gamma : discount factor
        self.gamma = params.get('gamma', 0.9)
        self.predict_mode = params.get('predict_mode', False)
        self.lrate = params.get('lrate', 0.0005)
        self.critw = params.get('critw', 2.)
        self.max_turn = params['max_turn'] + 4
        self.state_dimension = 2 * self.act_cardinality + 7 * self.slot_cardinality + 3 + self.max_turn
        self.build_shared_model()
        self.n = params.get('n', 5)

        ## load a model if present
        if params['trained_model_path'] != None:
            self.shared_model = copy.deepcopy(self.load(params['trained_model_path']))
            self.predict_mode = True
            self.warm_start = 2

    def load(self, name):
        self.shared_model.load(name)

    def save(self, name):
        self.shared_model.save_weights(name)

    def build_shared_model(self):
        print ('Building Shared A2C Architechture....')
        model = Sequential()
        fc1 = Dense(80, input_shape=(self.state_dimension,), activation='relu',
                    kernel_initializer=VarianceScaling(mode='fan_avg',
                                                       distribution='normal'), kernel_regularizer=regularizers.l2(self.reg_cost))
        fc2 = Dense(self.hidden_size, activation='relu',
                    kernel_initializer=VarianceScaling(mode='fan_avg',
                                                       distribution='normal'), kernel_regularizer=regularizers.l2(self.reg_cost))
        
        model.add(fc1)
        # model.add(fc2)
        
        actor_model = Dense(self.num_actions, activation='softmax',kernel_initializer=VarianceScaling(mode='fan_avg',
                                          distribution='normal'), kernel_regularizer=regularizers.l2(self.reg_cost),
                                          name='act_output')
        
        critic_model = Dense(1, activation='linear', kernel_initializer=VarianceScaling(mode='fan_avg',
                                          distribution='normal'), kernel_regularizer=regularizers.l2(self.reg_cost),
                                          name='crit_output')
        
        state_input = keras.layers.Input(shape=(self.state_dimension,), name='state_input')
        shared_output = model(state_input)
        actor_output = actor_model(shared_output)
        critic_output = critic_model(shared_output)
        
        shared_model = keras.models.Model(inputs=state_input, outputs=[actor_output, critic_output])
        shared_model.compile(optimizer=Adam(lr=self.lrate),
              loss={'act_output': 'categorical_crossentropy', 'crit_output': 'mean_squared_error'},
              loss_weights={'act_output': 1., 'crit_output': self.critw})
        
        self.shared_model = shared_model

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

    def get_epsilon(self, update_counter):
        if self.eps_strat == 'linear_decay':
            eps_threshold = self.eps_start + (self.eps_end - self.eps_start) * min((update_counter / self.eps_decay), 1)
        elif self.eps_strat == 'exp_decay':
            eps_decay = self.eps_decay / 100
            eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * update_counter / eps_decay)
        elif self.eps_strat == 'log_decay':
            # eps_threshold = eps_end + (eps_start - eps_end)
            # max(self.epsilon_min, min(self.epsilon, 1.0 - math.log10((t + 1) * self.epsilon_decay)))
            raise NotImplementedError()
        return eps_threshold

    def state_to_action(self, state):
        """ A2C: Input state, output action """
        ## Dialogue manager calls this to fill the experience buffer ##
        representation = self.prepare_state_representation(state)
        representation = np.expand_dims(np.asarray(representation), axis=0)
        self.action, _ = self.shared_model.predict(representation)
        self.action = self.action.squeeze(0)
        if self.eps_fixed == True:
            idx = np.random.choice(self.num_actions, 1, p=self.action)[0]
        else:
            # epsilon greedy with the epsilon declining  from 0.95 to 0
            if random.random() <= self.epsilon:
                idx = random.randint(0, self.num_actions - 1)
            else:
                idx = np.argmax(self.action)
        act_slot_response = copy.deepcopy(
            self.feasible_actions[idx])
        return {'act_slot_response': act_slot_response, 'act_slot_value_response': None}, idx, self.action[idx]

    def action_index(self, act_slot_response):
        """ Return the index of action """

        for (i, action) in enumerate(self.feasible_actions):
            if act_slot_response == action:
                return i
        print act_slot_response
        raise Exception("action index not found")
        return None

    def get_advantage(self, states, rewards):
        T = len(rewards)
        v_end = np.zeros(T)
        gain = np.zeros(T)
        advantage = np.zeros(T)
        # states = [self.prepare_state_representation(x) for x in states]
        for t in reversed(range(len(rewards) - 1)):
            if t + self.n >= T:
                v_end[t] = 0
            else:
                v_end[t] = self.shared_model.predict(np.asarray([states[t + self.n]]))[1][0][0]
            gain[t] = self.gamma ** self.n * v_end[t] + \
                      sum([(self.gamma ** k) * rewards[t + k] \
                               if t + k < T \
                               else self.gamma ** k * 0 \
                           for k in range(self.n)])
            advantage[t] = gain[t] - self.shared_model.predict(np.asarray([states[t]]))[1][0][0]
        return advantage, gain

    def train(self, states, actions, rewards, indexes, update_counter, gamma=0.99):

        self.epsilon = self.get_epsilon(update_counter)
        print("Epsilon: {0}".format(self.epsilon))


        states = [self.prepare_state_representation(x) for x in states]
        ## range for rewards in dialogue is reduced
        # rewards = [r*40 for r in rewards]
        advantage, gains = self.get_advantage(states, rewards)
        advantage = advantage.reshape(-1, 1)
        actions = np.asarray(actions)

        # L(\theta) from the handout
        targets = advantage  # * actions
        act_target = np.zeros((len(states), self.num_actions))
        act_target[np.arange(len(states)), np.array(indexes)] \
            = targets.squeeze(1)
        states = np.asarray(states)
        rewards = np.asarray(rewards)
        tot_rewards = np.sum(rewards)
        
        self.shared_model.train_on_batch(states, {'act_output': act_target,'crit_output': gains})
        return tot_rewards

    def evaluate(self, env, episode, num_episodes=100, render=False):

        cumulative_rewards = []
        for e in range(num_episodes):
            state = env.reset()
            tot_reward = 0
            while True:
                action_probs,_ = self.shared_model.predict(np.asarray([state]))
                action = np.random.choice(np.arange(
                    len(action_probs[0])), p=action_probs[0])
                state, reward, done, _ = env.step(action)
                tot_reward += reward
                if done:
                    break
            cumulative_rewards.append(tot_reward)
        mean_rewards = np.mean(cumulative_rewards)
        std_rewards = np.std(cumulative_rewards)
        return mean_rewards, std_rewards