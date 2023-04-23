from collections import deque
import argparse
import os
import time, datetime
import torch
import numpy as np
import pickle as pkl

from procgen import ProcgenEnv
from vec_env import VecExtractDictObs
from vec_env import VecMonitor
from vec_env import VecNormalize
from util import logger

from policies import ImpalaCNN
from ppo import PPO
from square import init_patch_square
from create_venv import create_venv
from evaluate import patch_evaluate_train
from evaluate import patch_evaluate_test
from evaluate import mask_train
from optimal_pos import optimal_pos
from optimal_pos import generate_buffer
from patch_pos import gradient_based_search



def parse_args():
    parser = argparse.ArgumentParser(
        description='Process procgen training arguments.')

    # Experiment parameters.
    parser.add_argument(
        '--distribution-mode', type=str, default='easy',
        choices=['easy', 'hard', 'exploration', 'memory', 'extreme'])
    parser.add_argument('--env_name', type=str, default='fruitbot')
    parser.add_argument('--num_envs', type=int, default= 4)
    parser.add_argument('--num-levels', type=int, default=200)
    parser.add_argument('--start-level', type=int, default=0)
    parser.add_argument('--num-threads', type=int, default= 1)
    parser.add_argument('--log-dir', type=str, default='./log')
    parser.add_argument('--model_file', type=str, default=None)

    # PPO parameters.
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num_episodes', type=int, default= 1000)
    parser.add_argument('--pos_episodes', type=int, default= 10000)
    parser.add_argument('--test_episodes', type=int, default= 10000)
    parser.add_argument('--train_episodes', type=int, default= 100000)
    parser.add_argument('--nsteps', type=int, default= 1000)
    parser.add_argument('--batch-size', type=int, default= 256)
    parser.add_argument('--lr', type = float, default = 5e-4)
    parser.add_argument('--ent-coef', type=float, default=0.01)
    parser.add_argument('--vf-coef', type=float, default=0.5)
    parser.add_argument('--gamma', type=float, default=0.999)
    parser.add_argument('--lam', type=float, default=0.95)
    parser.add_argument('--clip-range', type=float, default=0.2)
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    parser.add_argument('--nepochs', type=int, default=1)
    parser.add_argument('--max-steps', type=int, default=25_000_000)
    parser.add_argument('--save-interval', type=int, default=100)
    parser.add_argument('--epsilon-end', type=float, default=None)

    parser.add_argument('--standard', dest='standard', action='store_true', help='evaluate standard performance')
    parser.add_argument('--deterministic', dest='deterministic', action='store_true', help='makes agent always big most likely action')
    parser.add_argument('--pgd', dest='pgd', action='store_true', help='evaluate under PGD attack')
    parser.add_argument('--gwc', dest='gwc', action='store_true', help='evaluate Greedy Worst-Case Reward')
    parser.set_defaults(deterministic= True, pgd=False, standard=False, gwc=False)
    
    parser.add_argument('--percent_image', type = float, default = 0.05)
    parser.add_argument('--num_patch', type = int, default =  1)
    parser.add_argument('--replay_size', type = int, default = 100000, metavar = 'RS', help = 'size of replay')
    parser.add_argument('--replay_batch', type = int, default = 256, metavar = 'RB', help = 'size of replay batch')
    parser.add_argument('--potential_nums', type = int, default = 1)
    parser.add_argument('--random_eps', type = float, default = 0.3)
    parser.add_argument('--roa_sample', type = int, default = 256)
    parser.add_argument('--roa_region', type = int, default = 10)
    return parser.parse_args()



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
        act_deterministically = configs.deterministic,
    )
    

    return ppo_agent, train_venv, valid_venv



if __name__ == '__main__':
    configs = parse_args()
    ppo_agent, train_venv, valid_venv = env_set(configs)
    
    openstr = 'patchs/' + configs.env_name + '_' + str(configs.percent_image)
    with open(openstr,'rb') as f:
    	patch = pkl.load(f)
    openstr = 'x_pos/' + configs.env_name + '_' + str(configs.percent_image)
    with open(openstr,'rb') as f:
    	x_pos = pkl.load(f)
    openstr = 'y_pos/' + configs.env_name + '_' + str(configs.percent_image)
    with open(openstr,'rb') as f:
    	y_pos = pkl.load(f)
    
    train_mean, train_std = patch_evaluate_train(configs, ppo_agent, train_venv, valid_venv, patch, x_pos, y_pos)
    eval_mean, eval_std = patch_evaluate_test(configs, ppo_agent, train_venv, valid_venv, patch, x_pos, y_pos)
    
    file1 = open("train_reward.txt", "a")		
    str_txt= "\n" + configs.env_name + " percent image:" + str(configs.percent_image)+ " average reward:" + str(train_mean) + " reward std:" + str(train_std)
    file1.write(str_txt)
    file1.close()
    
    file2 = open("eval_reward.txt", "a")		
    str_txt= "\n" + configs.env_name + " percent image:" + str(configs.percent_image)+ " average reward:" + str(eval_mean) + " reward std:" + str(eval_std)
    file2.write(str_txt)
    file2.close()
    
  
