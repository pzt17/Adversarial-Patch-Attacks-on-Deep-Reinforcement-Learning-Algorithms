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
from evaluate import patch_evaluate
from evaluate import patch_train
from evaluate import patch_evaluate_test
from evaluate import train_test
from evaluate import eval_test
from optimal_pos import optimal_pos
from optimal_pos import generate_buffer
from safe_mean import safe_mean


def parse_args():
    parser = argparse.ArgumentParser(
        description='Process procgen training arguments.')

    # Experiment parameters.
    parser.add_argument(
        '--distribution-mode', type=str, default='easy',
        choices=['easy', 'hard', 'exploration', 'memory', 'extreme'])
    parser.add_argument('--env_name', type=str, default='fruitbot')
    parser.add_argument('--num-envs', type=int, default= 1)
    parser.add_argument('--num-levels', type=int, default=200)
    parser.add_argument('--start-level', type=int, default=0)
    parser.add_argument('--num-threads', type=int, default= 1)
    parser.add_argument('--log-dir', type=str, default='./log')
    parser.add_argument('--model-file', type=str, default=None)
    parser.add_argument('--patch_c', type=int, default= 10)

    # PPO parameters.
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num_episodes', type=int, default= 1000)
    parser.add_argument('--pos_episodes', type=int, default= 30000)
    parser.add_argument('--test_episodes', type=int, default= 3000)
    parser.add_argument('--nsteps', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default= 32)

    parser.add_argument('--standard', dest='standard', action='store_true', help='evaluate standard performance')
    parser.add_argument('--deterministic', dest='deterministic', action='store_true', help='makes agent always big most likely action')
    parser.add_argument('--pgd', dest='pgd', action='store_true', help='evaluate under PGD attack')
    parser.add_argument('--gwc', dest='gwc', action='store_true', help='evaluate Greedy Worst-Case Reward')
    parser.set_defaults(deterministic= True, pgd=False, standard=False, gwc=False)
    
    parser.add_argument('--percent_image', type = float, default = 0.05)
    parser.add_argument('--num_patch', type = int, default =  3)
    parser.add_argument('--replay_size', type = int, default = 100000, metavar = 'RS', help = 'size of replay')
    parser.add_argument('--replay_batch', type = int, default = 256, metavar = 'RB', help = 'size of replay batch')
    parser.add_argument('--potential_nums', type = int, default = 1)
    parser.add_argument('--random_eps', type = float, default = 0.3)
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
        act_deterministically = configs.deterministic
    )
    

    return ppo_agent, train_venv, valid_venv
    
    

if __name__ == '__main__':
    configs = parse_args()
    optimal_patch = init_patch_square(configs.percent_image)
    ppo_agent, train_venv, valid_venv = env_set(configs)
    
    replay_buffer = generate_buffer(configs, ppo_agent, train_venv, valid_venv)
    
    train_reward = []
    eval_reward = []
    for patch_cnt in range(configs.patch_c):
    	pos_x_list = []
    	pos_y_list = []
    	for patch_idx in range(configs.num_patch):
    		pos_x, pos_y = optimal_pos(configs, optimal_patch, ppo_agent, train_venv, valid_venv, replay_buffer)
    		pos_x_list.append(pos_x)
    		pos_y_list.append(pos_y)
    	
    	optimal_reward = 1000000.0
    	optimal_std = 0.0
    	optimal_x = 1
    	optimal_y = 1
    	for patch_idx in range(configs.num_patch):
    		print('patch cnt:', patch_cnt)
    		print('patch idx:', patch_idx)
    		
    		patch = init_patch_square(configs.percent_image)
    		patch[:,:,:] = 128.0
    		patch = patch_train(configs, ppo_agent, train_venv, valid_venv, patch, pos_x_list[patch_idx], pos_y_list[patch_idx], replay_buffer)
    	
    		patch_mean, patch_std = patch_evaluate(configs, ppo_agent, train_venv, valid_venv, patch, pos_x_list[patch_idx], pos_y_list[patch_idx])
    		if patch_mean < optimal_reward:
    			optimal_reward = patch_mean
    			optimal_std = patch_std
    			optimal_patch = patch
    			optimal_x = pos_x_list[patch_idx]
    			optimal_y = pos_y_list[patch_idx]
    		
    	popen = 'patchs/' + configs.env_name + "/" + str(configs.percent_image)  + "_" + str(patch_cnt)
    	with open(popen, 'wb') as f:
    		pkl.dump(optimal_patch, f)
	
    	pattackx = 'x_pos/' + configs.env_name + "/" + str(configs.percent_image)  + "_" + str(patch_cnt)
    	with open(pattackx, 'wb') as f1:
    		pkl.dump(optimal_x, f1)
	
    	pattacky = 'y_pos/' + configs.env_name + "/" + str(configs.percent_image)  + "_" + str(patch_cnt)
    	with open(pattacky, 'wb') as f:
    		pkl.dump(optimal_y, f)	
    		
    	train_tem_reward = train_test(configs, ppo_agent, train_venv, valid_venv, optimal_patch, optimal_x, optimal_y)
    	eval_tem_reward = eval_test(configs, ppo_agent, train_venv, valid_venv, optimal_patch, optimal_x, optimal_y)
    	train_reward.extend(train_tem_reward)
    	eval_reward.extend(eval_tem_reward)
    	
    train_mean = format(safe_mean(train_reward),".2f")
    train_std = format(np.std(train_reward),".2f")
    eval_mean = format(safe_mean(eval_reward),".2f")
    eval_std = format(np.std(eval_reward),".2f")
   
    file1 = open("train_reward.txt", "a")		
    str_txt= "\n" + configs.env_name + " percent image:" + str(configs.percent_image)+ " average reward:" + str(train_mean) + " reward std:" + str(train_std)
    file1.write(str_txt)
    file1.close()
   
    file2 = open("eval_reward.txt", "a")		
    str_txt2 = "\n" + configs.env_name + " percent image:" + str(configs.percent_image)+ " average reward:" + str(eval_mean) + " reward std:" + str(eval_std)
    file2.write(str_txt2)
    file2.close()
    
  
