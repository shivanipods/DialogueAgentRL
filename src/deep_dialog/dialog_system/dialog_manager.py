"""
Created on May 17, 2016

@author: xiul, t-zalipt
"""

import json
from . import StateTracker
from deep_dialog import dialog_config
import ipdb
from enum import Enum

def enum(*sequential, **named):
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)

Reward = enum('NORMAL', 'A2C', 'PAPER', 'LEXICAL', 'NO_PENALTY')

class DialogManager:
    """ A dialog manager to mediate the interaction between an agent and a customer """
    
    def __init__(self, agent, user, act_set, slot_set, movie_dictionary, 
            is_a2c=False, reward_function_idx=Reward.LEXICAL):
        self.agent = agent
        self.user = user
        self.act_set = act_set
        self.slot_set = slot_set
        self.state_tracker = StateTracker(act_set, slot_set, movie_dictionary)
        self.user_action = None
        self.reward = 0
        self.episode_over = False
        self.is_a2c = is_a2c
        self.reward_function_use = reward_function_idx

    def initialize_episode(self):
        """ Refresh state for new dialog """
        self.reward = 0
        self.episode_over = False
        self.state_tracker.initialize_episode()
        self.user_action = self.user.initialize_episode()
        self.state_tracker.update(user_action = self.user_action)
        
        if dialog_config.run_mode < 3:
            print ("New episode, user goal:")
            print json.dumps(self.user.goal, indent=2)
        self.print_function(user_action = self.user_action)
        self.agent.initialize_episode()

    def register_agent_action(self, state, agent_action, record_training_data=True):
        ########################################################################
        #   Update STATE and ACTION
        ########################################################################
        self.state = state
        self.agent_action = agent_action
        ########################################################################
        #   Register AGENT action with the state_tracker
        ########################################################################
        self.state_tracker.update(agent_action=self.agent_action)

        self.agent.add_nl_to_action(self.agent_action)  # add NL to Agent Dia_Act
        self.print_function(agent_action=self.agent_action['act_slot_response'])

        ########################################################################
        #   CALL USER TO TAKE HER TURN
        ########################################################################
        self.sys_action = self.state_tracker.dialog_history_dictionaries()[-1]
        self.user_action, self.episode_over, dialog_status = self.user.next(self.sys_action)
        if self.reward_function_use == Reward.NORMAL:
            print "Running Normal Reward..."
            self.reward = self.reward_function(dialog_status)
        elif self.reward_function_use == Reward.A2C:
            print "Running A2C Reward..."
            self.reward = self.reward_function_a2c(dialog_status)
        elif self.reward_function_use == Reward.PAPER:
            print "Running Paper Implementation Reward..."
            self.reward = self.reward_function_paper(dialog_status)
        elif self.reward_function_use == Reward.NO_PENALTY:
            print "Running Paper No Penalty Reward..."
            self.reward = self.reward_function_without_penalty(dialog_status)
        elif self.reward_function_use == Reward.LEXICAL:
            print "Running LEXICAL Reward..."
            self.reward = self.reward_function_lexical(dialog_status)

        ########################################################################
        #   Update state tracker with latest user action
        ########################################################################
        if self.episode_over != True:
            self.state_tracker.update(user_action=self.user_action)

        return (self.episode_over, self.reward)

    def next_turn(self, record_training_data=True):
        """ This function initiates each subsequent exchange between agent and user (agent first) """
        
        ########################################################################
        #   CALL AGENT TO TAKE HER TURN
        ########################################################################

        self.state = self.state_tracker.get_state_for_agent() ## this code is tracking the dialogue state
        self.agent_action, idx, actions = self.agent.state_to_action(self.state)
        
        ########################################################################
        #   Register AGENT action with the state_tracker
        ########################################################################
        self.state_tracker.update(agent_action=self.agent_action)
        
        self.agent.add_nl_to_action(self.agent_action) # add NL to Agent Dia_Act
        self.print_function(agent_action = self.agent_action['act_slot_response'])
        
        ########################################################################
        #   CALL USER TO TAKE HER TURN
        ########################################################################
        self.sys_action = self.state_tracker.dialog_history_dictionaries()[-1]
        self.user_action, self.episode_over, dialog_status = self.user.next(self.sys_action)
        if self.reward_function_use == 'normal':
            self.reward = self.reward_function(dialog_status)
        elif self.reward_function_use == 'a2c':
            self.reward = self.reward_function_a2c(dialog_status)
        elif self.reward_function_use == 'paper':
            self.reward = self.reward_function_paper(dialog_status)
        elif self.reward_function_use == 'lexical':
            self.reward = self.reward_function_lexical(dialog_status, self.sys_action)
        ########################################################################
        #   Update state tracker with latest user action
        ########################################################################
        if self.episode_over != True:
            self.state_tracker.update(user_action = self.user_action)
            self.print_function(user_action = self.user_action)

        ########################################################################
        #  Inform agent of the outcome for this timestep (s_t, a_t, r, s_{t+1}, episode_over)
        ########################################################################
        # No Experience Replay in A2C
        if self.is_a2c==False and record_training_data:
            self.agent.register_experience_replay_tuple(self.state, self.agent_action, self.reward, self.state_tracker.get_state_for_agent(), self.episode_over)
        
        return (self.episode_over, self.reward), idx, actions

    def reward_function(self, dialog_status):
        """ Reward Function 1: a reward function based on the dialog_status """
        if dialog_status == dialog_config.FAILED_DIALOG:
            reward = -self.user.max_turn #10
        elif dialog_status == dialog_config.SUCCESS_DIALOG:
            reward = 3*self.user.max_turn #20
        else:
            reward = -1
        return reward

    def reward_function_a2c(self, dialog_status):
        """ Reward Function 3: a reward function based on the dialog_status """
        if dialog_status == dialog_config.FAILED_DIALOG:
            reward = -0.025 #10
        elif dialog_status == dialog_config.SUCCESS_DIALOG:
            reward = 2 - 0.025
        else:
            reward = -0.025
        return reward
    
    def reward_function_paper(self, dialog_status):
        """ Reward Function 1: a reward function based on the dialog_status """
        if dialog_status == dialog_config.FAILED_DIALOG:
            reward = -self.user.max_turn #10
            if self.is_a2c:
                reward *= 0.05
        elif dialog_status == dialog_config.SUCCESS_DIALOG:
            if self.is_a2c:
                reward = 1 - 0.05 * self.user.max_turn
            else:
                reward = 2*self.user.max_turn #20
        else:
            reward = -1
        return reward    

    def reward_function_without_penalty(self, dialog_status):
        """ Reward Function 2: a reward function without penalty on per turn and failure dialog """
        if dialog_status == dialog_config.FAILED_DIALOG:
            reward = 0
        elif dialog_status == dialog_config.SUCCESS_DIALOG:
            reward = 2 * self.user.max_turn
        else:
            reward = 0
        return reward
 
    def reward_function_lexical(self, dialog_status, action):
        """ Reward Function 2: a reward function without penalty on per turn and failure dialog """
        if dialog_status == dialog_config.FAILED_DIALOG:
            reward = - 0.05 * self.user.max_turn
        elif dialog_status == dialog_config.SUCCESS_DIALOG:
            reward = 2 * self.user.max_turn
        else:
            reward = len(action['request_slots'].keys())
        return reward
       
    def print_function(self, agent_action=None, user_action=None):
        """ Print Function """
        if agent_action:
            if dialog_config.run_mode == 0:
                if self.agent.__class__.__name__ != 'AgentCmd':
                    print ("Turn %d sys: %s" % (agent_action['turn'], agent_action['nl']))
            elif dialog_config.run_mode == 1:
                if self.agent.__class__.__name__ != 'AgentCmd':
                    print("Turn %d sys: %s, inform_slots: %s, request slots: %s" % (agent_action['turn'], agent_action['diaact'], agent_action['inform_slots'], agent_action['request_slots']))
            elif dialog_config.run_mode == 2: # debug mode
                print("Turn %d sys: %s, inform_slots: %s, request slots: %s" % (agent_action['turn'], agent_action['diaact'], agent_action['inform_slots'], agent_action['request_slots']))
                print ("Turn %d sys: %s" % (agent_action['turn'], agent_action['nl']))
            
            if dialog_config.auto_suggest == 1:
                print('(Suggested Values: %s)' % (self.state_tracker.get_suggest_slots_values(agent_action['request_slots'])))
        elif user_action:
            if dialog_config.run_mode == 0:
                print ("Turn %d usr: %s" % (user_action['turn'], user_action['nl']))
            elif dialog_config.run_mode == 1: 
                print ("Turn %s usr: %s, inform_slots: %s, request_slots: %s" % (user_action['turn'], user_action['diaact'], user_action['inform_slots'], user_action['request_slots']))
            elif dialog_config.run_mode == 2: # debug mode, show both
                print ("Turn %d usr: %s, inform_slots: %s, request_slots: %s" % (user_action['turn'], user_action['diaact'], user_action['inform_slots'], user_action['request_slots']))
                print ("Turn %d usr: %s" % (user_action['turn'], user_action['nl']))
            
            if self.agent.__class__.__name__ == 'AgentCmd': # command line agent
                user_request_slots = user_action['request_slots']
                if 'ticket'in user_request_slots.keys(): del user_request_slots['ticket']
                if len(user_request_slots) > 0:
                    possible_values = self.state_tracker.get_suggest_slots_values(user_action['request_slots'])
                    for slot in possible_values.keys():
                        if len(possible_values[slot]) > 0:
                            print('(Suggested Values: %s: %s)' % (slot, possible_values[slot]))
                        elif len(possible_values[slot]) == 0:
                            print('(Suggested Values: there is no available %s)' % (slot))
                else:
                    kb_results = self.state_tracker.get_current_kb_results()
                    print ('(Number of movies in KB satisfying current constraints: %s)' % len(kb_results))
                    
