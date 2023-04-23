from collections import deque
import argparse
import os
import time, datetime
import torch
import numpy as np

from procgen import ProcgenEnv
from vec_env import VecExtractDictObs
from vec_env import VecMonitor
from vec_env import VecNormalize
from util import logger

import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F
from torch.autograd import Variable

from policies import ImpalaCNN
from ppo import PPO
from square import init_patch_square
from roll import rollout_one_step
from safe_mean import safe_mean
from upd_obs import upd_obs
from upd_obs import patch_obs
from upd_obs import update_patch
from upd_obs import mask_obs
from upd_obs import syn_obs
from upd_obs import syn_obs_eval
from qlay import Qlay
from qlay import state_lay
import copy
import random

def gau_noise(train_obs, per_std):
    noise = np.random.normal(0, per_std, (train_obs.shape[0], 64, 64, 3))
    train_obs = np.clip(train_obs + noise, 0, 255)
    return np.float32(train_obs)
    

def patch_pos(ori_state, gen_state, patch_len):
	ori_state = torch.tensor(ori_state)
	gen_state = torch.tensor(gen_state)

	rec_side_num = 64 - patch_len + 1
	rec_sum = torch.zeros(rec_side_num * rec_side_num)
	diff = torch.abs(ori_state - gen_state)
	
	#print(diff.shape)
	diff = torch.sum(diff, dim = 3)
	
	m = nn.AvgPool2d(kernel_size = patch_len, stride = 1)
	m2 = nn.MaxPool2d(kernel_size = rec_side_num, stride = 1, return_indices = True)
			
	#print(diff.shape)
	mx_val = m(diff)
	_, mx_idx = m2(mx_val)
	mx_idx = mx_idx.squeeze(0).squeeze(0).squeeze(0)
	
	x_pos = mx_idx // rec_side_num
	y_pos = mx_idx % rec_side_num
	
	return x_pos, y_pos
	
def np_patch(states_ori, region_size, x_pos_list, y_pos_list):
	batch_size = states_ori.shape[0]
	states = copy.deepcopy(states_ori)
	half_size = int(batch_size/2)
	
	x_pos = np.random.randint(0, 84 - region_size + 1, half_size)
	y_pos = np.random.randint(0, 84 - region_size + 1, half_size)
	for i in range(half_size):
		states[i, 0, x_pos[i]: x_pos[i] + region_size, y_pos[i]:y_pos[i] + region_size] = 255.0
			
	list_size = y_pos_list.shape[0]
	y_r = np.random.randint(0, list_size, half_size)
	x_r = np.random.randint(0, list_size, half_size)
	
	y_pos2 = y_pos_list[y_r]
	x_pos2 = x_pos_list[x_r]
	for i in range(half_size):
		states[half_size + i, 0, x_pos2[i]: x_pos2[i] + region_size, y_pos2[i]:y_pos2[i] + region_size] = 255.0
			
	return states, x_pos, y_pos, x_pos2, y_pos2

def gau_train(config, agent, train_env, test_env, patch_size, x_pos_list, y_pos_list, gen_model2, pgd_epsilon=None, gwc_epsilon=None):
    
    assert not (pgd_epsilon and gwc_epsilon)
    if config.gpu >= 0:
        device = torch.device("cuda:{}".format(config.gpu))
    else:
        device = torch.device("cpu")
    agent.model.load_from_file(config.model_file, device)
    
    train_epinfo_buf = []
    train_obs = train_env.reset()
    train_steps = np.zeros(config.num_envs, dtype=int)

    
    step_cnt = 0
    while step_cnt<config.train_episodes:
        step_cnt += 1
        
        assert agent.training
   
        
        train_obs, train_steps, train_epinfo = rollout_one_step(agent=agent, env=train_env, obs=train_obs, steps=train_steps, pgd_epsilon=pgd_epsilon, gwc_epsilon=gwc_epsilon)
        train_epinfo_buf.extend(train_epinfo)
        
        if (step_cnt) % config.nsteps == 0:
        	print('num_episodes:', step_cnt)
        	print('eplenmean:', safe_mean([info['r'] for info in train_epinfo_buf]))
    
    eval_rewards = [info['r'] for info in train_epinfo_buf]
    return agent, safe_mean(eval_rewards), np.std(eval_rewards)/len(eval_rewards)**0.5 
    
def gau_train_test(config, agent, train_env, test_env, per_std, pgd_epsilon=None, gwc_epsilon=None):
    
    assert not (pgd_epsilon and gwc_epsilon)
    if config.gpu >= 0:
        device = torch.device("cuda:{}".format(config.gpu))
    else:
        device = torch.device("cpu")
    
    train_epinfo_buf = []
    train_obs = train_env.reset()
    train_steps = np.zeros(config.num_envs, dtype=int)

    
    step_cnt = 0
    while step_cnt<config.test_episodes:
        step_cnt += 1
        
        with agent.eval_mode():
        	assert not agent.training
        
       		train_obs = gau_noise(train_obs, per_std)
        	train_obs, train_steps, train_epinfo = rollout_one_step(agent=agent, env=train_env, obs=train_obs, steps=train_steps, pgd_epsilon=pgd_epsilon, gwc_epsilon=gwc_epsilon)
        	train_epinfo_buf.extend(train_epinfo)
        
       		if (step_cnt) % config.nsteps == 0:
        		print('num_episodes:', step_cnt)
        		print('eplenmean:', safe_mean([info['r'] for info in train_epinfo_buf]))
    
    eval_rewards = [info['r'] for info in train_epinfo_buf]
    return safe_mean(eval_rewards), np.std(eval_rewards)/len(eval_rewards)**0.5 
        	
def train_distri(config, agent, train_env, patch, pos_x, pos_y, gen_model, pgd_epsilon=None, gwc_epsilon=None):
    
    assert not (pgd_epsilon and gwc_epsilon)
    if config.gpu >= 0:
        device = torch.device("cuda:{}".format(config.gpu))
    else:
        device = torch.device("cpu")
    
    
    train_epinfo_buf = []
    train_obs = train_env.reset()
    train_steps = np.zeros(config.num_envs, dtype=int)

    
    step_cnt = 0
    while step_cnt<config.test_episodes:
        step_cnt += 1
        
        with agent.eval_mode():
        	assert not agent.training
        	
        	train_obs = upd_obs(train_obs, patch, pos_x, pos_y)
        	
        	
        	train_obs = np.swapaxes(train_obs, 1, 3)
        	train_obs = np.swapaxes(train_obs, 2, 3)
        	train_obs_ten = torch.tensor(train_obs, requires_grad = False).to(device)
        	gen_obs = gen_model(train_obs_ten)
        	gen_obs = gen_obs.detach().cpu().numpy()
        	gen_obs = np.swapaxes(gen_obs, 1, 2)
        	train_obs = np.swapaxes(gen_obs, 2, 3)
        	
        	train_obs, train_steps, train_epinfo = rollout_one_step(agent=agent, env=train_env, obs=train_obs, steps=train_steps, pgd_epsilon=pgd_epsilon, gwc_epsilon=gwc_epsilon)
        	train_epinfo_buf.extend(train_epinfo)
        
        	if (step_cnt) % config.nsteps == 0:
        		print('num_episodes:', step_cnt)
        		print('eplenmean:', safe_mean([info['r'] for info in train_epinfo_buf]))
    
    eval_rewards = [info['r'] for info in train_epinfo_buf]
    return safe_mean(eval_rewards), np.std(eval_rewards)/len(eval_rewards)**0.5 
    
def train_distri(config, agent, train_env, patch, pos_x, pos_y, gen_model, pgd_epsilon=None, gwc_epsilon=None):
    
    assert not (pgd_epsilon and gwc_epsilon)
    if config.gpu >= 0:
        device = torch.device("cuda:{}".format(config.gpu))
    else:
        device = torch.device("cpu")
    
    
    train_epinfo_buf = []
    train_obs = train_env.reset()
    train_steps = np.zeros(config.num_envs, dtype=int)

    
    step_cnt = 0
    while step_cnt<config.test_episodes:
        step_cnt += 1
        
        with agent.eval_mode():
        	assert not agent.training
        	
        	train_obs = upd_obs(train_obs, patch, pos_x, pos_y)
        	
        	
        	train_obs = np.swapaxes(train_obs, 1, 3)
        	train_obs = np.swapaxes(train_obs, 2, 3)
        	train_obs_ten = torch.tensor(train_obs, requires_grad = False).to(device)
        	gen_obs = gen_model(train_obs_ten)
        	gen_obs = gen_obs.detach().cpu().numpy()
        	gen_obs = np.swapaxes(gen_obs, 1, 2)
        	train_obs = np.swapaxes(gen_obs, 2, 3)
        	
        	train_obs, train_steps, train_epinfo = rollout_one_step(agent=agent, env=train_env, obs=train_obs, steps=train_steps, pgd_epsilon=pgd_epsilon, gwc_epsilon=gwc_epsilon)
        	train_epinfo_buf.extend(train_epinfo)
        
        	if (step_cnt) % config.nsteps == 0:
        		print('num_episodes:', step_cnt)
        		print('eplenmean:', safe_mean([info['r'] for info in train_epinfo_buf]))
    
    eval_rewards = [info['r'] for info in train_epinfo_buf]
    return safe_mean(eval_rewards), np.std(eval_rewards)/len(eval_rewards)**0.5 
    
def eval_distri(config, agent, valid_env, patch, pos_x, pos_y, gen_model, pgd_epsilon=None, gwc_epsilon=None):
    
    assert not (pgd_epsilon and gwc_epsilon)
    if config.gpu >= 0:
        device = torch.device("cuda:{}".format(config.gpu))
    else:
        device = torch.device("cpu")
    
    
    train_epinfo_buf = []
    train_obs = valid_env.reset()
    train_steps = np.zeros(config.num_envs, dtype=int)

    
    step_cnt = 0
    while step_cnt<config.test_episodes:
        step_cnt += 1
        
        with agent.eval_mode():
        	assert not agent.training
        	
        	train_obs = upd_obs(train_obs, patch, pos_x, pos_y)
        	
        	
        	train_obs = np.swapaxes(train_obs, 1, 3)
        	train_obs = np.swapaxes(train_obs, 2, 3)
        	train_obs_ten = torch.tensor(train_obs, requires_grad = False).to(device)
        	gen_obs = gen_model(train_obs_ten)
        	gen_obs = gen_obs.detach().cpu().numpy()
        	gen_obs = np.swapaxes(gen_obs, 1, 2)
        	train_obs = np.swapaxes(gen_obs, 2, 3)
        	
        	train_obs, train_steps, train_epinfo = rollout_one_step(agent=agent, env=valid_env, obs=train_obs, steps=train_steps, pgd_epsilon=pgd_epsilon, gwc_epsilon=gwc_epsilon)
        	train_epinfo_buf.extend(train_epinfo)
        
        	if (step_cnt) % config.nsteps == 0:
        		print('num_episodes:', step_cnt)
        		print('eplenmean:', safe_mean([info['r'] for info in train_epinfo_buf]))
    
    eval_rewards = [info['r'] for info in train_epinfo_buf]
    return safe_mean(eval_rewards), np.std(eval_rewards)/len(eval_rewards)**0.5 

def rec_train_distri(config, agent, train_env, patch, pos_x, pos_y, gen_model, mask_model, pgd_epsilon=None, gwc_epsilon=None):
    
    assert not (pgd_epsilon and gwc_epsilon)
    if config.gpu >= 0:
        device = torch.device("cuda:{}".format(config.gpu))
    else:
        device = torch.device("cpu")
    
    
    train_epinfo_buf = []
    train_obs = train_env.reset()
    train_steps = np.zeros(config.num_envs, dtype=int)

    
    step_cnt = 0
    x_pos = pos_x
    y_pos = pos_y
    print(pos_x)
    print(pos_y)
    patch_len = patch.shape[1]
    rec_side_num = 64 - patch_len + 1
    rec_sum = torch.zeros(rec_side_num * rec_side_num)
    cx_pos = 0
    cy_pos = 0
    while step_cnt<config.test_episodes:
        step_cnt += 1
        
        with agent.eval_mode():
        	assert not agent.training
        	
        	
        	
        	train_obs = upd_obs(train_obs, patch, pos_x, pos_y)
        	origin_obs = copy.deepcopy(train_obs)
        	
        	train_obs = np.swapaxes(train_obs, 1, 3)
        	train_obs = np.swapaxes(train_obs, 2, 3)
        	train_obs_ten = torch.tensor(train_obs, requires_grad = False).to(device)
        	gen_obs = gen_model(train_obs_ten)
        	gen_obs = gen_obs.detach().cpu().numpy()
        	gen_obs = np.swapaxes(gen_obs, 1, 2)
        	gen_obs = np.swapaxes(gen_obs, 2, 3)
        	
        	if step_cnt% 1 == 0:
        		cx_pos, cy_pos = patch_pos(origin_obs, gen_obs, patch_len)
        		
        	masked_obs = mask_obs(origin_obs, patch, x_pos, y_pos)
        	masked_obs = np.swapaxes(masked_obs, 1, 3)
        	masked_obs = np.swapaxes(masked_obs, 2, 3)
        	masked_obs_ten = torch.tensor(masked_obs, requires_grad = False).to(device)
        	gen_obs2 = mask_model(masked_obs_ten)
        	gen_obs2 = gen_obs2.detach().cpu().numpy()
        	gen_obs2 = np.swapaxes(gen_obs2, 1, 2)
        	gen_obs2 = np.swapaxes(gen_obs2, 2, 3)
        	train_obs = syn_obs_eval(origin_obs, gen_obs2, patch_len, x_pos, y_pos)
        	
        	
        	train_obs, train_steps, train_epinfo = rollout_one_step(agent=agent, env=train_env, obs=train_obs, steps=train_steps, pgd_epsilon=pgd_epsilon, gwc_epsilon=gwc_epsilon)
        	train_epinfo_buf.extend(train_epinfo)
        
        	if (step_cnt) % config.nsteps == 0:
        		print('num_episodes:', step_cnt)
        		print('eplenmean:', safe_mean([info['r'] for info in train_epinfo_buf]))
    
    eval_rewards = [info['r'] for info in train_epinfo_buf]
    return eval_rewards
    
def rec_eval_distri(config, agent, valid_env, patch, pos_x, pos_y, gen_model, mask_model, pgd_epsilon=None, gwc_epsilon=None):
    
    assert not (pgd_epsilon and gwc_epsilon)
    if config.gpu >= 0:
        device = torch.device("cuda:{}".format(config.gpu))
    else:
        device = torch.device("cpu")
    
    
    train_epinfo_buf = []
    train_obs = valid_env.reset()
    train_steps = np.zeros(config.num_envs, dtype=int)

    
    step_cnt = 0
    x_pos = pos_x
    y_pos = pos_y
    print(pos_x)
    print(pos_y)
    patch_len = patch.shape[1]
    rec_side_num = 64 - patch_len + 1
    rec_sum = torch.zeros(rec_side_num * rec_side_num)
    cx_pos = 0
    cy_pos = 0
    while step_cnt<config.test_episodes:
        step_cnt += 1
        
        with agent.eval_mode():
        	assert not agent.training
        	
        	train_obs = upd_obs(train_obs, patch, pos_x, pos_y)
        	origin_obs = copy.deepcopy(train_obs)
        	
        	train_obs = np.swapaxes(train_obs, 1, 3)
        	train_obs = np.swapaxes(train_obs, 2, 3)
        	train_obs_ten = torch.tensor(train_obs, requires_grad = False).to(device)
        	gen_obs = gen_model(train_obs_ten)
        	gen_obs = gen_obs.detach().cpu().numpy()
        	gen_obs = np.swapaxes(gen_obs, 1, 2)
        	gen_obs = np.swapaxes(gen_obs, 2, 3)
        	
        	if step_cnt% 1 == 0:
        		cx_pos, cy_pos = patch_pos(origin_obs, gen_obs, patch_len)
        		
        	masked_obs = mask_obs(origin_obs, patch, x_pos, y_pos)
        	
        	
        	masked_obs = np.swapaxes(masked_obs, 1, 3)
        	masked_obs = np.swapaxes(masked_obs, 2, 3)
        	masked_obs_ten = torch.tensor(masked_obs, requires_grad = False).to(device)
        	gen_obs2 = mask_model(masked_obs_ten)
        	gen_obs2 = gen_obs2.detach().cpu().numpy()
        	gen_obs2 = np.swapaxes(gen_obs2, 1, 2)
        	gen_obs2 = np.swapaxes(gen_obs2, 2, 3)
        	train_obs = syn_obs_eval(origin_obs, gen_obs2, patch_len, x_pos, y_pos)
        	
        	
        	train_obs, train_steps, train_epinfo = rollout_one_step(agent=agent, env=valid_env, obs=train_obs, steps=train_steps, pgd_epsilon=pgd_epsilon, gwc_epsilon=gwc_epsilon)
        	train_epinfo_buf.extend(train_epinfo)
        
        	if (step_cnt) % config.nsteps == 0:
        		print('num_episodes:', step_cnt)
        		print('eplenmean:', safe_mean([info['r'] for info in train_epinfo_buf]))
    
    eval_rewards = [info['r'] for info in train_epinfo_buf]
    return eval_rewards
    
def rec_train_distri(config, agent, train_env, patch, pos_x, pos_y, gen_model, mask_model, pgd_epsilon=None, gwc_epsilon=None):
    
    assert not (pgd_epsilon and gwc_epsilon)
    if config.gpu >= 0:
        device = torch.device("cuda:{}".format(config.gpu))
    else:
        device = torch.device("cpu")
    
    
    train_epinfo_buf = []
    train_obs = train_env.reset()
    train_steps = np.zeros(config.num_envs, dtype=int)

    
    step_cnt = 0
    x_pos = pos_x
    y_pos = pos_y
    print(pos_x)
    print(pos_y)
    patch_len = patch.shape[1]
    rec_side_num = 64 - patch_len + 1
    rec_sum = torch.zeros(rec_side_num * rec_side_num)
    cx_pos = 0
    cy_pos = 0
    while step_cnt<config.test_episodes:
        step_cnt += 1
        
        with agent.eval_mode():
        	assert not agent.training
        	
        	train_obs = upd_obs(train_obs, patch, pos_x, pos_y)
        	origin_obs = copy.deepcopy(train_obs)
        	
        	train_obs = np.swapaxes(train_obs, 1, 3)
        	train_obs = np.swapaxes(train_obs, 2, 3)
        	train_obs_ten = torch.tensor(train_obs, requires_grad = False).to(device)
        	gen_obs = gen_model(train_obs_ten)
        	gen_obs = gen_obs.detach().cpu().numpy()
        	gen_obs = np.swapaxes(gen_obs, 1, 2)
        	gen_obs = np.swapaxes(gen_obs, 2, 3)
        	
        	if step_cnt% 1 == 0:
        		cx_pos, cy_pos = patch_pos(origin_obs, gen_obs, patch_len)
        		
        	masked_obs = mask_obs(origin_obs, patch, x_pos, y_pos)
        	masked_obs = np.swapaxes(masked_obs, 1, 3)
        	masked_obs = np.swapaxes(masked_obs, 2, 3)
        	masked_obs_ten = torch.tensor(masked_obs, requires_grad = False).to(device)
        	gen_obs2 = mask_model(masked_obs_ten)
        	gen_obs2 = gen_obs2.detach().cpu().numpy()
        	gen_obs2 = np.swapaxes(gen_obs2, 1, 2)
        	gen_obs2 = np.swapaxes(gen_obs2, 2, 3)
        	train_obs = syn_obs_eval(origin_obs, gen_obs2, patch_len, x_pos, y_pos)
        	
        	
        	train_obs, train_steps, train_epinfo = rollout_one_step(agent=agent, env=train_env, obs=train_obs, steps=train_steps, pgd_epsilon=pgd_epsilon, gwc_epsilon=gwc_epsilon)
        	train_epinfo_buf.extend(train_epinfo)
        
        	if (step_cnt) % config.nsteps == 0:
        		print('num_episodes:', step_cnt)
        		print('eplenmean:', safe_mean([info['r'] for info in train_epinfo_buf]))
    
    eval_rewards = [info['r'] for info in train_epinfo_buf]
    return eval_rewards
    
def random_patch_evaluate_train(config, agent, train_env, patch, pos_x, pos_y, gen_model, mask_model, pgd_epsilon=None, gwc_epsilon=None):
    
    assert not (pgd_epsilon and gwc_epsilon)
    if config.gpu >= 0:
        device = torch.device("cuda:{}".format(config.gpu))
    else:
        device = torch.device("cpu")
    
    
    train_epinfo_buf = []
    train_obs = train_env.reset()
    train_steps = np.zeros(config.num_envs, dtype=int)

    
    step_cnt = 0
    x_pos = pos_x
    y_pos = pos_y
    print(pos_x)
    print(pos_y)
    patch_len = patch.shape[1]
    rec_side_num = 64 - patch_len + 1
    rec_sum = torch.zeros(rec_side_num * rec_side_num)
    cx_pos = 0
    cy_pos = 0
    while step_cnt<config.test_episodes:
        step_cnt += 1
        
        with agent.eval_mode():
        	assert not agent.training
        	
        	pos_x = np.random.randint(0, 64 - patch_len + 1)
        	pos_y = np.random.randint(0, 64 - patch_len + 1)
        	
        	train_obs = upd_obs(train_obs, patch, pos_x, pos_y)
        	origin_obs = copy.deepcopy(train_obs)
        	
        	train_obs = np.swapaxes(train_obs, 1, 3)
        	train_obs = np.swapaxes(train_obs, 2, 3)
        	train_obs_ten = torch.tensor(train_obs, requires_grad = False).to(device)
        	gen_obs = gen_model(train_obs_ten)
        	gen_obs = gen_obs.detach().cpu().numpy()
        	gen_obs = np.swapaxes(gen_obs, 1, 2)
        	gen_obs = np.swapaxes(gen_obs, 2, 3)
        	
        	if step_cnt% 1 == 0:
        		cx_pos, cy_pos = patch_pos(origin_obs, gen_obs, patch_len)
        		
        	masked_obs = mask_obs(origin_obs, patch, x_pos, y_pos)
        	masked_obs = np.swapaxes(masked_obs, 1, 3)
        	masked_obs = np.swapaxes(masked_obs, 2, 3)
        	masked_obs_ten = torch.tensor(masked_obs, requires_grad = False).to(device)
        	gen_obs2 = mask_model(masked_obs_ten)
        	gen_obs2 = gen_obs2.detach().cpu().numpy()
        	gen_obs2 = np.swapaxes(gen_obs2, 1, 2)
        	gen_obs2 = np.swapaxes(gen_obs2, 2, 3)
        	train_obs = syn_obs_eval(origin_obs, gen_obs2, patch_len, x_pos, y_pos)
        	
        	
        	train_obs, train_steps, train_epinfo = rollout_one_step(agent=agent, env=train_env, obs=train_obs, steps=train_steps, pgd_epsilon=pgd_epsilon, gwc_epsilon=gwc_epsilon)
        	train_epinfo_buf.extend(train_epinfo)
        
        	if (step_cnt) % config.nsteps == 0:
        		print('num_episodes:', step_cnt)
        		print('eplenmean:', safe_mean([info['r'] for info in train_epinfo_buf]))
    
    eval_rewards = [info['r'] for info in train_epinfo_buf]
    return safe_mean(eval_rewards), np.std(eval_rewards)/len(eval_rewards)**0.5 
    
def random_patch_evaluate_test(config, agent, valid_env, patch, pos_x, pos_y, gen_model, mask_model, pgd_epsilon=None, gwc_epsilon=None):
    
    assert not (pgd_epsilon and gwc_epsilon)
    if config.gpu >= 0:
        device = torch.device("cuda:{}".format(config.gpu))
    else:
        device = torch.device("cpu")
    
    
    train_epinfo_buf = []
    train_obs = valid_env.reset()
    train_steps = np.zeros(config.num_envs, dtype=int)

    
    step_cnt = 0
    x_pos = pos_x
    y_pos = pos_y
    print(pos_x)
    print(pos_y)
    patch_len = patch.shape[1]
    rec_side_num = 64 - patch_len + 1
    rec_sum = torch.zeros(rec_side_num * rec_side_num)
    cx_pos = 0
    cy_pos = 0
    while step_cnt<config.test_episodes:
        step_cnt += 1
        
        with agent.eval_mode():
        	assert not agent.training
        	
        	pos_x = np.random.randint(0, 64 - patch_len + 1)
        	pos_y = np.random.randint(0, 64 - patch_len + 1)
        	
        	train_obs = upd_obs(train_obs, patch, pos_x, pos_y)
        	origin_obs = copy.deepcopy(train_obs)
        	
        	train_obs = np.swapaxes(train_obs, 1, 3)
        	train_obs = np.swapaxes(train_obs, 2, 3)
        	train_obs_ten = torch.tensor(train_obs, requires_grad = False).to(device)
        	gen_obs = gen_model(train_obs_ten)
        	gen_obs = gen_obs.detach().cpu().numpy()
        	gen_obs = np.swapaxes(gen_obs, 1, 2)
        	gen_obs = np.swapaxes(gen_obs, 2, 3)
        	
        	if step_cnt% 1 == 0:
        		cx_pos, cy_pos = patch_pos(origin_obs, gen_obs, patch_len)
        		
        	masked_obs = mask_obs(origin_obs, patch, x_pos, y_pos)
        	
        	
        	masked_obs = np.swapaxes(masked_obs, 1, 3)
        	masked_obs = np.swapaxes(masked_obs, 2, 3)
        	masked_obs_ten = torch.tensor(masked_obs, requires_grad = False).to(device)
        	gen_obs2 = mask_model(masked_obs_ten)
        	gen_obs2 = gen_obs2.detach().cpu().numpy()
        	gen_obs2 = np.swapaxes(gen_obs2, 1, 2)
        	gen_obs2 = np.swapaxes(gen_obs2, 2, 3)
        	train_obs = syn_obs_eval(origin_obs, gen_obs2, patch_len, x_pos, y_pos)
        	
        	
        	train_obs, train_steps, train_epinfo = rollout_one_step(agent=agent, env=valid_env, obs=train_obs, steps=train_steps, pgd_epsilon=pgd_epsilon, gwc_epsilon=gwc_epsilon)
        	train_epinfo_buf.extend(train_epinfo)
        
        	if (step_cnt) % config.nsteps == 0:
        		print('num_episodes:', step_cnt)
        		print('eplenmean:', safe_mean([info['r'] for info in train_epinfo_buf]))
    
    eval_rewards = [info['r'] for info in train_epinfo_buf]
    return safe_mean(eval_rewards), np.std(eval_rewards)/len(eval_rewards)**0.5 
