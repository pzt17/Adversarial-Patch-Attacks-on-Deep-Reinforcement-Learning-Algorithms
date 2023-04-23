from collections import deque
import argparse
import os
import time, datetime
import torch
import torch.nn as nn
import numpy as np
import pickle as pkl
import copy
import random

from procgen import ProcgenEnv
from vec_env import VecExtractDictObs
from vec_env import VecMonitor
from vec_env import VecNormalize
from util import logger

from policies import ImpalaCNN
from ppo import PPO
from square import init_patch_square
from create_venv import create_venv
from evaluate import patch_evaluate
from evaluate import patch_train
from evaluate import patch_evaluate_test
from optimal_pos import optimal_pos
from optimal_pos import generate_buffer
from generator import Generator


def parse_args():
    parser = argparse.ArgumentParser(
        description='Process procgen training arguments.')

    # Experiment parameters.
    parser.add_argument(
        '--distribution-mode', type=str, default='easy',
        choices=['easy', 'hard', 'exploration', 'memory', 'extreme'])
    parser.add_argument('--env_name', type=str, default='fruitbot')
    parser.add_argument('--num-envs', type=int, default= 8)
    parser.add_argument('--num-levels', type=int, default=200)
    parser.add_argument('--start-level', type=int, default=0)
    parser.add_argument('--num-threads', type=int, default= 1)
    parser.add_argument('--log-dir', type=str, default='./log')
    parser.add_argument('--model-file', type=str, default=None)

    # PPO parameters.
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num_episodes', type=int, default= 30000)
    parser.add_argument('--pos_episodes', type=int, default= 20000)
    parser.add_argument('--nsteps', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default= 64)

    parser.add_argument('--standard', dest='standard', action='store_true', help='evaluate standard performance')
    parser.add_argument('--deterministic', dest='deterministic', action='store_true', help='makes agent always big most likely action')
    parser.add_argument('--pgd', dest='pgd', action='store_true', help='evaluate under PGD attack')
    parser.add_argument('--gwc', dest='gwc', action='store_true', help='evaluate Greedy Worst-Case Reward')
    parser.set_defaults(deterministic= True, pgd=False, standard= True, gwc=False)
    
    parser.add_argument('--percent_image', type = float, default = 0.05)
    parser.add_argument('--num_patch', type = int, default =  1)
    parser.add_argument('--replay_size', type = int, default = 100000, metavar = 'RS', help = 'size of replay')
    parser.add_argument('--replay_batch', type = int, default = 8, metavar = 'RB', help = 'size of replay batch')
    parser.add_argument('--potential_nums', type = int, default = 1)
    parser.add_argument('--random_eps', type = float, default = 0.3)
    parser.add_argument("--lr", type=float, default=0.01, help="adam: learning rate")
    return parser.parse_args()

def patch_obs(states, patch_size):
	batch_size = states.shape[0]
	states = copy.deepcopy(states)
	gau = np.random.rand(patch_size, patch_size, 3)*255.0
	
	pos_x = np.random.randint(0, 64 - patch_size + 1, batch_size)
	pos_y = np.random.randint(0, 64 - patch_size + 1, batch_size)
	
	for i in range(batch_size):
		for rgb in range(3):
			states[i, rgb, pos_x[i]:pos_x[i] + patch_size, pos_y[i]:pos_y[i] + patch_size] = random.random()*255.0
	return states, pos_x, pos_y


def env_set(configs):
    random_seed = 0
    train_venv = create_venv(configs, is_valid=False, seed=random_seed)
    valid_venv = create_venv(configs, is_valid=True, seed=random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    policy = ImpalaCNN(
        obs_space=train_venv.observation_space,
        num_outputs=train_venv.action_space.n,
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=0, eps=1e-5)
    
    ppo_agent = PPO(
        model=policy,
        optimizer=optimizer,
        gpu=configs.gpu,
        minibatch_size=configs.batch_size,
        act_deterministically = configs.deterministic
    )
    

    return ppo_agent, train_venv, valid_venv
    
    

if __name__ == '__main__':
    configs = parse_args()
    optimal_patch = init_patch_square(configs.percent_image)
    ppo_agent, train_venv, valid_venv = env_set(configs)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    replay_buffer = generate_buffer(configs, ppo_agent, train_venv, valid_venv)
    
    gen_model = Generator()
    gen_model.to(device)
    optimizer = torch.optim.Adam(gen_model.parameters(), lr=configs.lr)
    ploss = nn.MSELoss()
    patch = init_patch_square(configs.percent_image)
    patch_size = patch.shape[1]
    tot_loss = 0.0
    ctot_loss = 0.0
    closs = nn.MSELoss()
    for frame in range(configs.num_episodes):
	
        if replay_buffer.__len__() >= configs.replay_batch:
        	optimizer.zero_grad()
        	states = replay_buffer.sample(configs.replay_batch)
        	states = np.swapaxes(states, 1, 3)
        	states = np.swapaxes(states, 2, 3)
        	origin_states = copy.deepcopy(states)
        	states, pos_x, pos_y = patch_obs(states, patch_size)
        	states = torch.tensor(np.float32(states), requires_grad = True).to(device)
        	out_states = gen_model(states)
        	origin_states = torch.tensor(np.float32(origin_states), requires_grad = False).to(device)
        	loss = ploss(origin_states, out_states) 
        	tot_loss = tot_loss + loss
        	loss.backward()
        	optimizer.step()
        	
        	if frame%500 == 0:
        		print('loss:', tot_loss/500.0)
        		tot_loss = 0.0

    pt_name = "base/" + "single_" + configs.env_name + "_" + str(configs.percent_image)
    torch.save(gen_model.state_dict(), pt_name)
	
  
    
  
