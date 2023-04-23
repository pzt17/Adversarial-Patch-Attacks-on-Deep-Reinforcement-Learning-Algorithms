import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F
from IPython.display import clear_output
import matplotlib.pyplot as plt
from environment import make_atari, wrap_deepmind, wrap_pytorch
import argparse

from dqn import DuelingCnnDQN
from square import init_patch_square
from utils import read_config
from qlay import state_lay

def add_relay(state_relay, num_frames, env, per_correct, model):
	state = env.reset()
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	
	for frame in range(num_frames):
		state = torch.FloatTensor(np.float32(state)).unsqueeze(0)
		state = state.to(device)
		state_relay.push(state.squeeze(0).detach().cpu().numpy())
		action = model.act_ran(state, per_correct) 
		next_state, reward, done, _ = env.step(action)
		env.render()
		state = next_state
		
		if done:
			state = env.reset()
	return 
	

