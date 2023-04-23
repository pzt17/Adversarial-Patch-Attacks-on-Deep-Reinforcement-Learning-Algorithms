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
from environment import make_atari, wrap_deepmind, wrap_pytorch
from patch_pos import gradient_based_search
from pgd_patch import pgd_patch

def patch_gen(patch, model, env, relay, relay_sample_size, img_size, width, height, xskip, yskip, potential_nums, relay_capacity, relay_batch, num_frames):
	roa_states = relay.sample(relay_sample_size)
	pos_y, pos_x = gradient_based_search(model,img_size, roa_states, width, height, xskip, yskip, potential_nums)
	patch = pgd_patch(patch, pos_x, pos_y, model, img_size, env, num_frames, relay_capacity, relay_batch, relay)
	#reward = patch_test(patch, pos_x, pos_y, model, img_size, env, num_frames)
	return patch, pos_x, pos_y
