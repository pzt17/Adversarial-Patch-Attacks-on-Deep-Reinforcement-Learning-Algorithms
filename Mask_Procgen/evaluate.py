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
from upd_obs import ran_mask_obs
from qlay import Qlay
from qlay import state_lay
import copy
from mstate_patch import mstate_patch

def patch_pos(ori_state, gen_state, patch_len):
	ori_state = torch.tensor(ori_state)
	gen_state = torch.tensor(gen_state)

	rec_side_num = 64 - patch_len + 1
	rec_sum = torch.zeros(rec_side_num * rec_side_num)
	diff = torch.abs(ori_state - gen_state)
	diff = torch.sum(diff, dim = 3)
	
	m = nn.AvgPool2d(kernel_size = patch_len, stride = 1)
	m2 = nn.MaxPool2d(kernel_size = rec_side_num, stride = 1, return_indices = True)
			
	mx_val = m(diff)
	_, mx_idx = m2(mx_val)
	mx_idx = mx_idx.squeeze(0).squeeze(0)
	
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
		states[i, 0, x_pos[i]: x_pos[i] + region_size, y_pos[i]:y_pos[i] + region_size] = 0.0
			
	list_size = y_pos_list.shape[0]
	y_r = np.random.randint(0, list_size, half_size)
	x_r = np.random.randint(0, list_size, half_size)
	
	y_pos = y_pos_list[y_r]
	x_pos = x_pos_list[x_r]
	for i in range(half_size):
		states[half_size + i, 0, x_pos[i]: x_pos[i] + region_size, y_pos[i]:y_pos[i] + region_size] = 0.0
			
	return states

def patch_evaluate_train(config, agent, train_env, test_env, patch, pos_x, pos_y, pgd_epsilon=None, gwc_epsilon=None):

    assert not (pgd_epsilon and gwc_epsilon)
    if config.gpu >= 0:
        device = torch.device("cuda:{}".format(config.gpu))
    else:
        device = torch.device("cpu")
    load_address = config.model_file + '_' + str(config.percent_image)
    agent.model.load_from_file(load_address, device)
    logger.info('Loaded model from {}.'.format(config.model_file))

    train_epinfo_buf = []
    train_obs = train_env.reset()
    train_steps = np.zeros(config.num_envs, dtype=int)


    step_cnt = 0
    while step_cnt<config.test_episodes:
        step_cnt += 1
        with agent.eval_mode():
            assert not agent.training
            
            train_obs = mask_obs(train_obs, patch, pos_x, pos_y)
            train_obs, train_steps, train_epinfo = rollout_one_step(agent=agent, env=train_env, obs=train_obs, steps=train_steps, pgd_epsilon=pgd_epsilon, gwc_epsilon=gwc_epsilon)
            train_epinfo_buf.extend(train_epinfo)

        if (step_cnt + 1) % config.nsteps == 0:
        	print('num_episodes:', step_cnt)
        	print('eplenmean:', safe_mean([info['r'] for info in train_epinfo_buf]))

    eval_rewards = [info['r'] for info in train_epinfo_buf]
    return safe_mean(eval_rewards), np.std(eval_rewards)/len(eval_rewards)**0.5
    
def patch_evaluate_test(config, agent, train_env, test_env, patch, pos_x, pos_y, pgd_epsilon=None, gwc_epsilon=None):

    assert not (pgd_epsilon and gwc_epsilon)
    if config.gpu >= 0:
        device = torch.device("cuda:{}".format(config.gpu))
    else:
        device = torch.device("cpu")
    load_address = config.model_file + '_' + str(config.percent_image)
    agent.model.load_from_file(load_address, device)
    logger.info('Loaded model from {}.'.format(config.model_file))

    train_epinfo_buf = []
    train_obs = test_env.reset()
    train_steps = np.zeros(config.num_envs, dtype=int)


    step_cnt = 0
    while step_cnt<config.test_episodes:
        step_cnt += 1
        with agent.eval_mode():
            assert not agent.training
            train_obs = mask_obs(train_obs, patch, pos_x, pos_y)
            train_obs, train_steps, train_epinfo = rollout_one_step(agent=agent, env=test_env, obs=train_obs, steps=train_steps, pgd_epsilon=pgd_epsilon, gwc_epsilon=gwc_epsilon)
            train_epinfo_buf.extend(train_epinfo)

        if (step_cnt + 1) % config.nsteps == 0:
        	print('num_episodes:', step_cnt)
        	print('eplenmean:', safe_mean([info['r'] for info in train_epinfo_buf]))

    eval_rewards = [info['r'] for info in train_epinfo_buf]
    return safe_mean(eval_rewards), np.std(eval_rewards)/len(eval_rewards)**0.5


    

def mask_train(config, agent, train_env, test_env, patch, x_pos_list, y_pos_list, pgd_epsilon=None, gwc_epsilon=None):
    
    assert not (pgd_epsilon and gwc_epsilon)
    if config.gpu >= 0:
        device = torch.device("cuda:{}".format(config.gpu))
    else:
        device = torch.device("cpu")
    agent.model.load_from_file(config.model_file, device)
    
    train_epinfo_buf = []
    train_obs = train_env.reset()
    train_steps = np.zeros(config.num_envs, dtype=int)
    patch_size = patch.shape[0]


    
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
        	
def rec_patch_evaluate_train(config, agent, train_env, test_env, patch, pos_x, pos_y, gen_model, pgd_epsilon=None, gwc_epsilon=None):

    assert not (pgd_epsilon and gwc_epsilon)
    if config.gpu >= 0:
        device = torch.device("cuda:{}".format(config.gpu))
    else:
        device = torch.device("cpu")
    load_address = config.model_file + '_' + str(config.percent_image)
    agent.model.load_from_file(load_address, device)
    logger.info('Loaded model from {}.'.format(config.model_file))

    train_epinfo_buf = []
    train_obs = train_env.reset()
    train_steps = np.zeros(config.num_envs, dtype=int)


    step_cnt = 0

    x_pos = pos_x
    y_pos = pos_y
    cx_pos = pos_x
    cy_pos = pos_y
    patch_len = patch.shape[0]
    pos_idx = 0
    rec_side_num = 64 - patch_len + 1
    rec_sum = torch.zeros(rec_side_num * rec_side_num)
    while step_cnt<config.test_episodes:
        step_cnt += 1
        with agent.eval_mode():
            assert not agent.training
            
            
            train_obs = upd_obs(train_obs, patch, pos_x, pos_y)
            pre_train_obs = copy.deepcopy(train_obs)
            
            
            if step_cnt % 1 == 0:
            	train_obs = np.swapaxes(train_obs, 1, 3)
            	train_obs = np.swapaxes(train_obs, 2, 3)
            	train_obs_ten = torch.tensor(train_obs, requires_grad = False).to(device)
            	gen_obs = gen_model(train_obs_ten)
            	gen_obs = gen_obs.detach().cpu().numpy()
            	gen_obs = np.swapaxes(gen_obs, 1, 2)
            	gen_obs = np.swapaxes(gen_obs, 2, 3)
            	
            	cx_pos, cy_pos = patch_pos(pre_train_obs, gen_obs, patch_len)
            
            train_obs = mask_obs(pre_train_obs, patch, x_pos, y_pos)
            
            
            train_obs, train_steps, train_epinfo = rollout_one_step(agent=agent, env=train_env, obs=train_obs, steps=train_steps, pgd_epsilon=pgd_epsilon, gwc_epsilon=gwc_epsilon)
            train_epinfo_buf.extend(train_epinfo)
            
            if step_cnt%1000 == 0:
            	print('step:', step_cnt)
            	print('x_pos', x_pos)
            	print('y_pos', y_pos)
            	rec_sum = torch.zeros(rec_side_num * rec_side_num)

        if (step_cnt + 1) % config.nsteps == 0:
        	print('num_episodes:', step_cnt)
        	print('eplenmean:', safe_mean([info['r'] for info in train_epinfo_buf]))

    eval_rewards = [info['r'] for info in train_epinfo_buf]
    return eval_rewards
    
def rec_patch_evaluate_test(config, agent, train_env, test_env, patch, pos_x, pos_y, gen_model, pgd_epsilon=None, gwc_epsilon=None):

    assert not (pgd_epsilon and gwc_epsilon)
    if config.gpu >= 0:
        device = torch.device("cuda:{}".format(config.gpu))
    else:
        device = torch.device("cpu")
    load_address = config.model_file + '_' + str(config.percent_image)
    agent.model.load_from_file(load_address, device)
    logger.info('Loaded model from {}.'.format(config.model_file))

    train_epinfo_buf = []
    train_obs = test_env.reset()
    train_steps = np.zeros(config.num_envs, dtype=int)


    step_cnt = 0

    x_pos = pos_x
    y_pos = pos_y
    cx_pos = pos_x
    cy_pos = pos_y
    patch_len = patch.shape[0]
    pos_idx = 0
    rec_side_num = 64 - patch_len + 1
    rec_sum = torch.zeros(rec_side_num * rec_side_num)
    while step_cnt<config.test_episodes:
        step_cnt += 1
        with agent.eval_mode():
            assert not agent.training
            
            
            train_obs = upd_obs(train_obs, patch, pos_x, pos_y)
            pre_train_obs = copy.deepcopy(train_obs)
            
            if step_cnt % 1 == 0:	
            	train_obs = np.swapaxes(train_obs, 1, 3)
            	train_obs = np.swapaxes(train_obs, 2, 3)
            	train_obs_ten = torch.tensor(train_obs, requires_grad = False).to(device)
            	gen_obs = gen_model(train_obs_ten)
            	gen_obs = gen_obs.detach().cpu().numpy()
            	gen_obs = np.swapaxes(gen_obs, 1, 2)
            	gen_obs = np.swapaxes(gen_obs, 2, 3)
            
            	cx_pos, cy_pos = patch_pos(pre_train_obs, gen_obs, patch_len)
            
            train_obs = mask_obs(pre_train_obs, patch, x_pos, y_pos)
            
            
            
            train_obs, train_steps, train_epinfo = rollout_one_step(agent=agent, env=test_env, obs=train_obs, steps=train_steps, pgd_epsilon=pgd_epsilon, gwc_epsilon=gwc_epsilon)
            train_epinfo_buf.extend(train_epinfo)
            
            if step_cnt%1000 == 0:
            	print('step:', step_cnt)
            	print('x_pos', x_pos)
            	print('y_pos', y_pos)
            	rec_sum = torch.zeros(rec_side_num * rec_side_num)

        if (step_cnt + 1) % config.nsteps == 0:
        	print('num_episodes:', step_cnt)
        	print('eplenmean:', safe_mean([info['r'] for info in train_epinfo_buf]))

    eval_rewards = [info['r'] for info in train_epinfo_buf]
    return eval_rewards
    
def random_patch_evaluate_train(config, agent, train_env, test_env, patch, pos_x, pos_y, pgd_epsilon=None, gwc_epsilon=None):

    assert not (pgd_epsilon and gwc_epsilon)
    if config.gpu >= 0:
        device = torch.device("cuda:{}".format(config.gpu))
    else:
        device = torch.device("cpu")
    load_address = config.model_file + '_' + str(config.percent_image)
    agent.model.load_from_file(load_address, device)
    logger.info('Loaded model from {}.'.format(config.model_file))

    train_epinfo_buf = []
    train_obs = train_env.reset()
    train_steps = np.zeros(config.num_envs, dtype=int)


    step_cnt = 0
    while step_cnt<config.test_episodes:
        step_cnt += 1
        with agent.eval_mode():
            assert not agent.training
            
            train_obs = ran_mask_obs(train_obs, patch)
            train_obs, train_steps, train_epinfo = rollout_one_step(agent=agent, env=train_env, obs=train_obs, steps=train_steps, pgd_epsilon=pgd_epsilon, gwc_epsilon=gwc_epsilon)
            train_epinfo_buf.extend(train_epinfo)

        if (step_cnt + 1) % config.nsteps == 0:
        	print('num_episodes:', step_cnt)
        	print('eplenmean:', safe_mean([info['r'] for info in train_epinfo_buf]))

    eval_rewards = [info['r'] for info in train_epinfo_buf]
    return safe_mean(eval_rewards), np.std(eval_rewards)/len(eval_rewards)**0.5
    
def random_patch_evaluate_test(config, agent, train_env, test_env, patch, pos_x, pos_y, pgd_epsilon=None, gwc_epsilon=None):

    assert not (pgd_epsilon and gwc_epsilon)
    if config.gpu >= 0:
        device = torch.device("cuda:{}".format(config.gpu))
    else:
        device = torch.device("cpu")
    load_address = config.model_file + '_' + str(config.percent_image)
    agent.model.load_from_file(load_address, device)
    logger.info('Loaded model from {}.'.format(config.model_file))

    train_epinfo_buf = []
    train_obs = test_env.reset()
    train_steps = np.zeros(config.num_envs, dtype=int)


    step_cnt = 0
    while step_cnt<config.test_episodes:
        step_cnt += 1
        with agent.eval_mode():
            assert not agent.training
            train_obs = ran_mask_obs(train_obs, patch)
            train_obs, train_steps, train_epinfo = rollout_one_step(agent=agent, env=test_env, obs=train_obs, steps=train_steps, pgd_epsilon=pgd_epsilon, gwc_epsilon=gwc_epsilon)
            train_epinfo_buf.extend(train_epinfo)

        if (step_cnt + 1) % config.nsteps == 0:
        	print('num_episodes:', step_cnt)
        	print('eplenmean:', safe_mean([info['r'] for info in train_epinfo_buf]))

    eval_rewards = [info['r'] for info in train_epinfo_buf]
    return safe_mean(eval_rewards), np.std(eval_rewards)/len(eval_rewards)**0.5

  
