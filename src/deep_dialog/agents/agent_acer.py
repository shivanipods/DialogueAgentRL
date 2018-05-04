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
from collections import namedtuple
from keras import regularizers
import ipdb
from constants import *
import random
import gym
import math

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


class AgentACER(Agent):
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

        # experience replay
        self.experience_replay_pool_size = params.get('experience_replay_pool_size', 1000)
        self.experience_replay_pool = []

        self.hidden_size = params.get('dqn_hidden_size', 60)
        self.predict_mode = params.get('predict_mode', False)
        self.actor_lr = params.get('actor_lr', 0.0005)
        self.critic_lr = params.get('critic_lr', 0.001)

        # hypoerparamter for experience replay actor critic
        # gamma : discount factor
        self.gamma = params.get('gamma', 0.99)
        self.clip = params.get('clip', 1)
        self.beta = params.get('beta', 0.99)
        self.lrate = params.get('lrate',  0.0005)

        ## epsilon parameters
        self.eps_fixed = params.get("eps_fixed", False)
        self.eps_strat = params.get("eps_start", "linear_decay")
        self.eps_start = params.get('eps_start', 0.3)
        self.eps_end = params.get('eps_end', 0)
        self.eps_decay = params.get('eps_decay', 1e3)


        ## warm start:
        self.warm_start = params.get('warm_start', 0)
        self.max_turn = params['max_turn'] + 4
        self.state_dimension = 2 * self.act_cardinality + 7 * self.slot_cardinality + 3 + self.max_turn

        self.build_common_model()
        self.ac_model_clone = keras.models.clone_model(self.ac_model)
        self.n = params.get('n', 50)

        ## load a model if present
        if params['trained_model_path'] != None:
            self.actor_model = copy.deepcopy(self.load(
                params['trained_actor_model_path']))
            self.critic_model = copy.deepcopy(self.load(
                params['trained_critic_model_path']))
            self.predict_mode = True
            self.warm_start = 2

    def load(self, name, model_name):
        if model_name == "actor":
            self.actor_model.load(name)
        else:
            self.critic_model.load(name)

    def save(self, name, model_name):
        if model_name == "actor":
            self.actor_model.save_weights(name)
        else:
            self.critic_model.save_weights(name)

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


    def build_actor_model(self):
        model = Sequential()
        fc1 = Dense(50, input_shape=(self.state_dimension,), activation='relu',
                    kernel_initializer=VarianceScaling(mode='fan_avg',
                                                       distribution='normal'), kernel_regularizer=regularizers.l2(self.reg_cost))
        fc2 = Dense(50, activation='relu',
                    kernel_initializer=VarianceScaling(mode='fan_avg',
                                                       distribution='normal'), kernel_regularizer=regularizers.l2(self.reg_cost))
        fc3 = Dense(self.num_actions, activation='softmax',
                    kernel_initializer=VarianceScaling(mode='fan_avg',
                                                       distribution='normal'), kernel_regularizer=regularizers.l2(self.reg_cost))
        model.add(fc1)
        model.add(fc2)
        model.add(fc3)
        model.compile(loss='mse', optimizer=Adam(lr=self.actor_lr))
        self.actor_model = model

    def build_critic_model(self):
        model = Sequential()
        fc1 = Dense(50, input_shape=(self.state_dimension,), activation='relu',
                    kernel_initializer=VarianceScaling(mode='fan_avg',
                                                       distribution='normal'), kernel_regularizer=regularizers.l2(self.reg_cost))
        fc2 = Dense(50, activation='relu',
                    kernel_initializer=VarianceScaling(mode='fan_avg',
                                                       distribution='normal'), kernel_regularizer=regularizers.l2(self.reg_cost))
        fc3 = Dense(1, activation='relu',
                    kernel_initializer=VarianceScaling(mode='fan_avg',
                                                       distribution='normal'), kernel_regularizer=regularizers.l2(self.reg_cost))
        model.add(fc1)
        model.add(fc2)
        model.add(fc3)
        model.compile(loss='mse', optimizer=Adam(lr=self.critic_lr))
        self.critic_model = model

    def build_common_model(self):
        shared_model = Sequential()
        fc1 = Dense(130, input_shape=(self.state_dimension,), activation='relu',
                    kernel_initializer=VarianceScaling(mode='fan_avg',
                                                       distribution='normal'),
                    kernel_regularizer=regularizers.l2(self.reg_cost))
        fc2 = Dense(50, activation='relu',
                    kernel_initializer=VarianceScaling(mode='fan_avg',
                                                       distribution='normal'),
                    kernel_regularizer=regularizers.l2(self.reg_cost))
        shared_model.add(fc1)
        shared_model.add(fc2)
        self.actor_model = Dense(self.num_actions, activation='softmax',
                    kernel_initializer=VarianceScaling(mode='fan_avg',
                                                       distribution='normal'), kernel_regularizer=regularizers.l2(self.reg_cost), name='act_output')
        self.critic_model = Dense(self.num_actions, activation='linear',
                    kernel_initializer=VarianceScaling(mode='fan_avg',
                                                       distribution='normal'), kernel_regularizer=regularizers.l2(self.reg_cost), name='crit_output')

        state_input = keras.layers.Input(shape=(self.state_dimension,), name='state_input')
        shared_output = shared_model(state_input)
        actor_output =self.actor_model(shared_output)
        critic_output = self.critic_model(shared_output)

        model = keras.models.Model(inputs=state_input, outputs=[actor_output, critic_output])
        optimizer_ = Adam(lr=self.lrate)
        model.compile(optimizer=optimizer_,
                           loss={'act_output': 'categorical_crossentropy', 'crit_output': 'mean_squared_error'},
                           loss_weights={'act_output': 1, 'crit_output': 1})
        self.ac_model = model

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

    def register_experience_replay_dialogue(self, states, actions, action_probs, rewards):
        ## no check for predict mode and warms_start parameters
        ## no q values beings stored
        training_example = (states, actions, action_probs, rewards)
        if len(self.experience_replay_pool) == self.experience_replay_pool_size:
            ## deque the least recent one
            self.experience_replay_pool.pop(0)
        self.experience_replay_pool.append(training_example)

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

    def state_to_action(self, state):
        """ A2C: Input state, output action """
        ## Dialogue manager calls this to fill the experience buffer ##
        representation = self.prepare_state_representation(state)
        representation = np.expand_dims(np.asarray(representation), axis=0)
        self.action, _ = self.ac_model.predict(representation)
        self.action = self.action.squeeze(0)

        if self.warm_start == 1:
            ## if in training mode(not prediction) fill until you cant anymore
            idx  = self.rule_policy()
            act_slot_response = copy.deepcopy(
                self.feasible_actions[idx])
            return {'act_slot_response': act_slot_response, 'act_slot_value_response': None}, idx, self.action[idx]
        if self.eps_fixed == True:
            idx = np.random.choice(self.num_actions, 1, p=self.action)[0]
        else:
            # epsilon greedy with the epsilon declining  from 0.95 to 0
            if random.random() <= self.epsilon:
                idx = random.randint(0, self.num_actions - 1)
            else:
                idx = np.argmax(self.action)
        ## multinomial distrbution
        # idx = np.random.choice(self.num_actions, 1, p=self.action)[0]
        act_slot_response = copy.deepcopy(
            self.feasible_actions[idx])
        return {'act_slot_response': act_slot_response, 'act_slot_value_response': None}, idx, self.action[idx]


    def train_batch(self, batch, batch_size):
        eps = 1e-6
        states = batch.states
        actions = batch.actions
        action_probs = batch.action_probs
        rewards = batch.rewards


    def train(self, batch, update_counter, gamma=0.99):
        ## data structures to collect things in batch for training

        ## decay epsilon for the next batch
        self.epsilon = self.get_epsilon(update_counter)
        print("Epsilon: {0}".format(self.epsilon))

        eps = 1e-6
        batch_states = []
        batch_advantages = []
        batch_gains = []
        batch_actions = []
        total_rewards = 0

        for dialogue in batch:
            states = dialogue[0]
            actions = dialogue[1]
            action_probs = dialogue[2]
            rewards = dialogue[3]
            total_rewards += np.sum(rewards)

            q_ret = 0
            advantages = np.zeros(len(rewards)) ## zero everywhere except where the actions actually took place
            values = np.zeros(len(rewards))
            importances = np.zeros(len(rewards))
            gains = np.zeros(len(rewards)) ## recompute q-function for same state representation and replace action values with gains

            for t in reversed(range(len(rewards) - 1)):
                state_representation = np.expand_dims(np.asarray(states[t]), axis=0)
                actor_values, q_values = self.ac_model.predict(state_representation)

                # importance ratio
                actor_values = actor_values.squeeze(0)
                importance = actor_values[actions[t]]/(action_probs[t] + eps)
                if np.isnan(importance):
                    importance = 1
                importances[t] = np.minimum(self.clip, importance)

                # value function
                values[t] = np.sum(actor_values * q_values)

                q_ret = rewards[t] + self.gamma*q_ret
                advantages[t] = (q_ret - values[t])*importances[t]

                gains[t] = q_ret

                # advantages are the targets for cross entropy for training actor
                # Q_ret currently computed is the correct q value for given state
                q_ret = importances[t]*(q_ret - q_values.squeeze(0)[actions[t]]) + values[t]

            batch_states.extend(states)
            batch_advantages.extend(advantages)
            batch_gains.extend(gains)
            batch_actions.extend(actions)

        # train on batch
        batch_size = len(batch_states)
        batch_states = np.asarray(batch_states)
        act_targets = np.zeros((batch_size, self.num_actions))
        act_targets[np.arange(batch_size), np.array(batch_actions)] = batch_advantages

        ## arrays*43 with the actions filled out
        _,crit_targets = self.ac_model.predict(batch_states)
        crit_targets[np.arange(batch_size), np.array(batch_actions)] = batch_gains

        metric, act_loss, crit_loss = self.ac_model.train_on_batch(batch_states, {'act_output': act_targets,
                                                                         'crit_output': crit_targets})

        ## interpolate the wights of the clone and actual model
        self.ac_model_clone.set_weights([(1-self.beta)*w for w in self.ac_model.get_weights()] +
                                        [self.beta*w for w in self.ac_model_clone.get_weights()])

        ## print loss
        print("Actor Loss: {0:4f}".format(act_loss))
        print("Critic Loss: {0:4f}".format(crit_loss))

        return float(total_rewards)/batch_size

    def evaluate(self, env, episode, num_episodes=100, render=False):

        cumulative_rewards = []
        for e in range(num_episodes):
            state = env.reset()
            tot_reward = 0
            while True:
                action_probs = self.actor_model.predict(np.asarray([state]))
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