import math, random
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F
from IPython.display import clear_output
import matplotlib.pyplot as plt
from torch.autograd import Variable
from environment import make_atari, wrap_deepmind, wrap_pytorch
from qlay import Qlay
import copy

def pgd_patch(patch, pos_x, pos_y, model, img_size, env, num_frames, relay_capacity, relay_batch, relay):
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	
	for frames in range(num_frames):
			states  = relay.sample(relay_batch)

			ori_states = copy.deepcopy(states)
			states = patch_states(states, patch, pos_x, pos_y)
			states = torch.FloatTensor(np.float32(states))
			states = states.to(device)
			states = Variable(states, requires_grad = True)
			ori_states = torch.FloatTensor(np.float32(ori_states))
			ori_states = ori_states.to(device)
			q_values = model.get_q_value(ori_states)
			q_values = Variable(q_values)
			ms = nn.Softmax(dim = 1)
			attacked_q_values = ms(model.get_q_value(states))
			loss = torch.sum(attacked_q_values * q_values)
			loss.backward()
			states_grad = states.grad.detach().cpu().numpy()
			
			patch = update_patch(states_grad, patch, pos_y, pos_x)
    		
	return patch
	
def patch_state(state, patch, pos_x, pos_y):
	patch_size = patch.shape[2]
	for i in range(patch_size):
		for j in range(patch_size):
			state[0, pos_y + i, pos_x + j] = patch[0,i,j]*(1.0/255.0)
	return state
	
def patch_states(states, patch, pos_x, pos_y):
	patch_size = patch.shape[2]
	states[:, 0, pos_y:pos_y + patch_size, pos_x:pos_x + patch_size] = patch[0,:,:]*(1.0/255.0);
	return states
	
def update_patch(states_grad, patch, pos_y, pos_x):
	grad_sum = np.zeros((states_grad.shape[1], states_grad.shape[2], states_grad.shape[3]))
	patch_size = patch.shape[2]
	grad_sum[0, pos_y: pos_y + patch_size, pos_x: pos_x + patch_size] += np.sum(states_grad[:, 0, pos_y: pos_y + patch_size, pos_x: pos_x + patch_size], axis = 0)
	
	patch[0,:,:] -= 1.0 * np.sign(grad_sum[0, pos_y: pos_y + patch_size, pos_x: pos_x + patch_size])
	patch = np.clip(patch, 0, 255)
	return patch
