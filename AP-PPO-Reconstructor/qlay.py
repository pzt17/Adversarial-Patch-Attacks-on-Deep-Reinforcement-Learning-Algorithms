import math, random
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F
from collections import deque

class state_lay(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state):
        #state = np.expand_dims(state, 0)
        self.buffer.append((state))
    
    def sample(self, batch_size):
        state  = random.sample(self.buffer, batch_size)
        
        #print(state[0].shape)
        #h = np.concatenate(state)
        #print('h', h.shape)
        return np.concatenate(state)
    
    def __len__(self):
        return len(self.buffer)
  
class Qlay(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, q_val):
        #state = np.expand_dims(state, 0)
        #q_val = np.expand_dims(q_val, 0)
        self.buffer.append((state, q_val))
    
    def sample(self, batch_size):
        state, q_val = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), np.concatenate(q_val)
    
    def __len__(self):
        return len(self.buffer)
  
