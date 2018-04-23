## implement a neural network that learns an implicit reward function on simulated episodes
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

def train_implicit_reward():
	# DQN 
	return


if __name__ == '__main__':
    train_implicit_reward()
