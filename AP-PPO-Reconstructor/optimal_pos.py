from collections import deque
import argparse
import os
import time, datetime
import torch
import torch.nn as nn
import numpy as np
import copy

from procgen import ProcgenEnv
from vec_env import VecExtractDictObs
from vec_env import VecMonitor
from vec_env import VecNormalize
from util import logger

from policies import ImpalaCNN
from ppo import PPO
from square import init_patch_square
from create_venv import create_venv
from roll import rollout_one_step
from roll import rollout_one_step_oppor
from safe_mean import safe_mean
from qlay import state_lay

import torch.autograd as autograd 
from torch.autograd import Variable


def generate_buffer(config, agent, train_env, test_env, pgd_epsilon=None, gwc_epsilon=None):
    assert not (pgd_epsilon and gwc_epsilon)
    if config.gpu >= 0:
        device = torch.device("cuda:{}".format(config.gpu))
    else:
        device = torch.device("cpu")
    agent.model.load_from_file(config.model_file, device)

    train_obs = train_env.reset()
    train_steps = np.zeros(config.num_envs, dtype=int)
    train_epinfo_buf = []

	
    replay_buffer = state_lay(config.replay_size)
    step_cnt = 0
    while step_cnt <config.pos_episodes:
        step_cnt += 1
        with agent.eval_mode():
        	train_obs, train_steps, train_epinfo = rollout_one_step(agent=agent, env=train_env, obs= train_obs, steps=train_steps, pgd_epsilon=pgd_epsilon, gwc_epsilon=gwc_epsilon)
        	train_epinfo_buf.extend(train_epinfo)
        	replay_buffer.push(train_obs)
        	
        	
        	if (step_cnt)%1000 == 0:
        		print('num_episodes:', step_cnt)
        		print('eplenmean:', safe_mean([info['r'] for info in train_epinfo_buf[:config.num_episodes]]))
    step_cnt = 0
    while step_cnt < config.pos_episodes:
        step_cnt += 1
        with agent.eval_mode():
        	train_obs, train_steps, train_epinfo = rollout_one_step_oppor(agent=agent, env=train_env, obs= train_obs, steps=train_steps, pgd_epsilon=pgd_epsilon, gwc_epsilon=gwc_epsilon, random_ep = config.random_eps,  num_envs = train_obs.shape[0])
        	train_epinfo_buf.extend(train_epinfo)
        	replay_buffer.push(train_obs)
        	
        	
        	if (step_cnt)%1000 == 0:
        		print('num_episodes:', step_cnt)
        		print('eplenmean:', safe_mean([info['r'] for info in train_epinfo_buf[:config.num_episodes]]))
        		
    return replay_buffer

def optimal_pos(configs, patch, ppo_agent, train_venv, valid_venv, replay_buffer):
    x_pos, y_pos = patch_pos(configs, ppo_agent, train_venv, valid_venv, patch, replay_buffer)
    return x_pos, y_pos


def patch_pos(config, agent, train_env, test_env, patch, replay_buffer, pgd_epsilon=None, gwc_epsilon=None):
    samples = replay_buffer.sample(config.replay_batch)
    patch_len = patch.shape[0]
    opt_x, opt_y = gradient_based_search(agent, 64, samples, patch_len, patch_len, 1, 1, config.potential_nums)
    
    print(opt_x)
    print(opt_y)
    
    return opt_x, opt_y
    
def gradient_based_search(model, size, states, width, height, xskip, yskip, potential_nums):
	with model.eval_mode():

        	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        	states_ori = copy.deepcopy(states)
        	states = torch.tensor(np.float32(states)).to(device)
        	states_grad = torch.tensor(np.float32(states_ori)).to(device)
        	states_grad = Variable(states_grad, requires_grad = True)
        	qvs = model.get_q_value(states)
        	qvs = Variable(qvs, requires_grad = False)
       		ms = nn.Softmax(dim = 1)
        	attacked_q_values = ms(model.get_q_value(states_grad))
        	loss = torch.sum(attacked_q_values * qvs)
        	loss.backward()
        
        	sum_grad = torch.sum(states_grad.grad, dim = 0)
        	sum_grad = torch.sum(sum_grad, dim = 2)
        	print('sum_grad shape:', sum_grad.shape)
        	mx_val, _ = torch.max(torch.abs(sum_grad.view(sum_grad.shape[0]*sum_grad.shape[1], -1)), dim = 0)
        	sum_grad = sum_grad / mx_val
        
        	xcnt = ((size - width)//xskip) + 1
        	ycnt = ((size - height)//yskip) + 1
        	losses = torch.zeros(xcnt*ycnt)
        	for i in range(xcnt):
        		for j in range(ycnt):
        			tm = sum_grad[i*xskip: i*xskip + width, j*yskip: j*yskip + height]
        			losses[i*xcnt + j] = torch.sum(torch.sum(torch.mul(tm,tm), 1))
        	tk_, tpk_idx = torch.topk(losses, potential_nums)
        	xidx = tpk_idx // xcnt
        	yidx = tpk_idx % xcnt
        
        	print(xidx)
        	print(yidx)
        
        	opt_x = 1
        	opt_y = 1
        	mx_loss = 100000000
        	with torch.set_grad_enabled(False):
        		for i in range(potential_nums):
        			patch_grey = states.clone()
        			patch_grey[:, xskip*xidx[i]:xskip*xidx[i] + width, yskip*yidx[i]:yskip*yidx[i] + height, :] = 128.0
        			cur_loss = torch.sum(ms(model.get_q_value(patch_grey))*qvs)
        			if cur_loss < mx_loss:
        				mx_loss = cur_loss
        				opt_x = xidx[i]*xskip
        				opt_y = yidx[i]*yskip
	
	return opt_x, opt_y
       
    

