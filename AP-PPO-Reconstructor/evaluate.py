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
from qlay import Qlay
from qlay import state_lay
import copy



def patch_evaluate(config, agent, train_env, test_env, patch, pos_x, pos_y, pgd_epsilon=None, gwc_epsilon=None):

    assert not (pgd_epsilon and gwc_epsilon)
    if config.gpu >= 0:
        device = torch.device("cuda:{}".format(config.gpu))
    else:
        device = torch.device("cpu")
    agent.model.load_from_file(config.model_file, device)
    logger.info('Loaded model from {}.'.format(config.model_file))

    train_epinfo_buf = []
    train_obs = train_env.reset()
    train_steps = np.zeros(config.num_envs, dtype=int)


    step_cnt = 0
    while step_cnt<config.test_episodes:
        step_cnt += 1
        with agent.eval_mode():
            assert not agent.training
            #patch2 = patch + np.random.randn(11, 11, 3)
            #patch2 = np.clip(patch2, 0, 255)
            train_obs = upd_obs(train_obs, patch2, pos_x, pos_y)
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
    agent.model.load_from_file(config.model_file, device)
    logger.info('Loaded model from {}.'.format(config.model_file))

    train_epinfo_buf = []
    train_obs = test_env.reset()
    train_steps = np.zeros(config.num_envs, dtype=int)


    step_cnt = 0
    while step_cnt<config.test_episodes:
        step_cnt += 1
        with agent.eval_mode():
            assert not agent.training
            train_obs = upd_obs(train_obs, patch, pos_x, pos_y)
            train_obs, train_steps, train_epinfo = rollout_one_step(agent=agent, env=test_env, obs=train_obs, steps=train_steps, pgd_epsilon=pgd_epsilon, gwc_epsilon=gwc_epsilon)
            train_epinfo_buf.extend(train_epinfo)

        if (step_cnt + 1) % config.nsteps == 0:
        	print('num_episodes:', step_cnt)
        	print('eplenmean:', safe_mean([info['r'] for info in train_epinfo_buf]))

    eval_rewards = [info['r'] for info in train_epinfo_buf]
    return safe_mean(eval_rewards), np.std(eval_rewards)/len(eval_rewards)**0.5

def patch_train(config, agent, train_env, test_env, patch, pos_x, pos_y, replay_buffer, pgd_epsilon=None, gwc_epsilon=None):
    
    assert not (pgd_epsilon and gwc_epsilon)
    if config.gpu >= 0:
        device = torch.device("cuda:{}".format(config.gpu))
    else:
        device = torch.device("cpu")
    agent.model.load_from_file(config.model_file, device)
    
    train_epinfo_buf = []
    train_obs = train_env.reset()
    train_steps = np.zeros(config.num_envs, dtype = int)
    
    step_cnt = 0
    ms = nn.Softmax(dim = 1)
    while step_cnt< config.num_episodes:
    	
    	step_cnt += 1
    	with agent.eval_mode():
            assert not agent.training
            
            if step_cnt%100 == 0:
            	print('episode:', step_cnt)
            	
            	
            if replay_buffer.__len__() >= config.replay_batch:
            	obs = replay_buffer.sample(config.replay_batch)
            	probs = copy.deepcopy(obs)
            	obs = patch_obs(obs, patch, pos_x, pos_y)
            	
            	obs = torch.FloatTensor(np.float32(obs)).to(device)
            	obs = Variable(obs, requires_grad = True)
            	
            	probs = torch.FloatTensor(np.float32(probs)).to(device)
            	probs = ms(agent.get_q_value(probs))
            	probs = Variable(probs)
            	attacked_probs = ms(agent.get_q_value(obs))
            	loss = torch.sum(attacked_probs * probs)
            	loss.backward()
            	
            	obs_grad = obs.grad.detach().cpu().numpy()
            	patch = update_patch(obs_grad, patch, pos_x, pos_y)
    
    print(patch)
    print(patch.shape)     
    return patch

