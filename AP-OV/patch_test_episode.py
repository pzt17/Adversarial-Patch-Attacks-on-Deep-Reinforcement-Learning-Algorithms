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
import torch.nn.functional as F
from torch.autograd import Variable

from wrappers import make_atari, wrap_deepmind, wrap_pytorch
from qlay import state_lay

def patch_test_episode(patch, pos_x, pos_y, model, img_size, env, num_frames):
	state = env.reset()
	episode_reward = 0.0
	episode_record = []
	episode_cnt = 0.0
	
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	
	for frames in range(num_frames):
		state = patch_state(state, patch, pos_x, pos_y)
		state = torch.FloatTensor(np.float32(state))
		state = state.unsqueeze(0)
		state = state.to(device)
		action = model.act(state)
		next_state, reward, done, _ = env.step(action)
		env.render()
		state = next_state
		episode_reward += reward
		
		if done:
        		state = env.reset()
        		print('Episode Reward in Testing Phase:', episode_reward)
        		episode_record.append(episode_reward)
        		episode_cnt += 1
        		episode_reward = 0.0
        		
	if episode_cnt == 0:
		episode_record.append(0.0)
		return episode_record
	else:
		return episode_record
	
def patch_state(state, patch, pos_x, pos_y):
	patch_size = patch.shape[2]
	state[0, pos_y: pos_y + patch_size, pos_x: pos_x + patch_size] = patch[0,:,:]*(1.0/255.0)
	return state
	

