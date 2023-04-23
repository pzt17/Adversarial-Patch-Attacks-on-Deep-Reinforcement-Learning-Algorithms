import math, random
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F
from collections import deque
from wrappers import make_atari, wrap_deepmind, wrap_pytorch
from IPython.display import clear_output
import matplotlib.pyplot as plt

class state_lay(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state):
        state = np.expand_dims(state, 0)
        self.buffer.append((state))
    
    def sample(self, batch_size):
        state  = random.sample(self.buffer, batch_size)
        return np.concatenate(state)
    
    def __len__(self):
        return len(self.buffer)
  
class Qlay(object):
    def __init__(self, capacity, num_action):
        self.buffer = []
        for i in range(num_action):
        	self.buffer.append([])
        self.capacity = capacity
        self.num_action = num_action
    
    def push(self, state, action):
        self.buffer[action].append(state)
        if len(self.buffer[action]) > self.capacity:
        	self.buffer[action].pop(0)
    
    def sample(self, batch_size):
        states =   np.array(random.sample(self.buffer[0], batch_size))
        for idx in range(1, self.num_action):
        	state = np.array(random.sample(self.buffer[idx], batch_size))
        	states = np.concatenate((states, state), axis = 0)
        return states
        
    def __len__(self):
        return len(self.buffer)
        
    def double_sample(self, batch_size):
    	state1 = self.sample(batch_size)
    	state2 = self.sample(batch_size)
    	states = np.concatenate((state1,state2), axis = 0)
    	return states
        
class Qlay2(object):
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity
    
    def push(self, state):
        self.buffer.append(state)
        if len(self.buffer) > self.capacity:
        	self.buffer.pop(0)
    
    def sample(self, batch_size):
        states =  random.sample(self.buffer, batch_size)
        return np.array(states)
        
    def __len__(self):
        return len(self.buffer)
  
  
