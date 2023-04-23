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
        
class mlay(object):
    def __init__(self, capacity, batch_size):
        self.buffer = deque(maxlen= batch_size)
        self.batch_size = batch_size
        self.batch_buffer = deque(maxlen = capacity)
    
    def push(self, state):
    	self.buffer.append((state))
    	state = np.expand_dims(self.buffer[-1],0)
    	for i in range(-2, -self.batch_size - 1, -1):
    		state = np.concatenate((state, np.expand_dims(self.buffer[i], 0)), axis = 0)
    	self.batch_buffer.append(state)
    
    def short_push(self, state):
    	self.buffer.append((state))
    	
    def sample(self):
        state  = random.sample(self.batch_buffer, 1)
        return np.array(state).squeeze(2)
    
    def __len__(self):
        return len(self.buffer)
       
    def deque_empty(self):
    	self.buffer.clear()
    
    def current(self):
    	return self.batch_buffer[-1].squeeze(1)
  
class Qlay(object):
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity
    
    def push(self, state):
        self.buffer.append(state)
        if len(self.buffer) > self.capacity:
        	self.buffer.pop(0)
    
    def sample(self, batch_size):
        state = random.sample(self.buffer, batch_size)
        return np.array(state)
        
    def __len__(self):
        return len(self.buffer)
  
