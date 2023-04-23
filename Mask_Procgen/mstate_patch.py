import collections
import itertools
import random
import math

import numpy as np
import torch
import torch.nn.functional as F
import copy



def mstate_patch(states_ori, region_size, x_pos_list, y_pos_list):
	gpu = 0
	device = torch.device("cuda:{}".format(gpu))
	
	batch_size = states_ori.shape[0]
	states = states_ori.clone()
	half_size = int(batch_size/2)
	
	x_pos = np.random.randint(0, 64 - region_size + 1, half_size)
	y_pos = np.random.randint(0, 64 - region_size + 1, half_size)
	x_pos = torch.tensor(x_pos).to(device)
	y_pos = torch.tensor(y_pos).to(device)
	for i in range(half_size):
		states[0: int(batch_size/2), x_pos[i]: x_pos[i] + region_size, y_pos[i]:y_pos[i] + region_size, :] = 255.0
			
	list_size = x_pos_list.shape[0]
	y_r = np.random.randint(0, list_size, half_size)
	x_r = np.random.randint(0, list_size, half_size)
	y_r = torch.tensor(y_r).to(device)
	x_r = torch.tensor(x_r).to(device)
	y_pos = y_pos_list[y_r]
	x_pos = x_pos_list[x_r]
	for i in range(half_size):
		states[int(batch_size/2):, x_pos[i]: x_pos[i] + region_size, y_pos[i]:y_pos[i] + region_size, :] = 255.0
			
	return states
    
