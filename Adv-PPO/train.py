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
from create_venv import create_venv
from evaluate import gau_train
from evaluate import gau_train_test
from evaluate import train_distri
from generator import Generator
from patch_pos import gradient_based_search
from optimal_pos import generate_buffer


def parse_args():
    parser = argparse.ArgumentParser(
        description='Process procgen training arguments.')

    # Experiment parameters.
    parser.add_argument(
        '--distribution-mode', type=str, default='easy',
        choices=['easy', 'hard', 'exploration', 'memory', 'extreme'])
    parser.add_argument('--env_name', type=str, default='fruitbot')
    parser.add_argument('--num_envs', type=int, default= 6)
    parser.add_argument('--num-levels', type=int, default=200)
    parser.add_argument('--start-level', type=int, default=0)
    parser.add_argument('--num-threads', type=int, default= 1)
    parser.add_argument('--log-dir', type=str, default='./log')
    parser.add_argument('--model_file', type=str, default=None)

    # PPO parameters.
    parser.add_argument('--gpu', type=int, default=0)
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
    

    return ppo_agent,  train_venv, valid_venv

def env_set2(configs, patch_size, x_pos_list, y_pos_list, gen_model2):
    random_seed = 0
    train_venv = create_venv(configs, is_valid=False, seed=random_seed)
    valid_venv = create_venv(configs, is_valid=True, seed=random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    policy = ImpalaCNN(
        obs_space=train_venv.observation_space,
        num_outputs=train_venv.action_space.n,
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr= configs.lr, eps=1e-5)
    
    ppo_agent = PPO(
        model=policy,
        optimizer=optimizer,
        gpu=configs.gpu,
        gamma=configs.gamma,
        lambd=configs.lam,
        value_func_coef=configs.vf_coef,
        entropy_coef=configs.ent_coef,
        update_interval=configs.nsteps * configs.num_envs,
        minibatch_size=configs.batch_size,
        epochs=configs.nepochs,
        clip_eps=configs.clip_range,
        clip_eps_vf=configs.clip_range,
        max_grad_norm=configs.max_grad_norm,
        epsilon_end=configs.epsilon_end,
        max_updates = configs.max_steps * configs.nepochs / configs.batch_size,
        patch_size = patch_size,
        x_pos_list = x_pos_list,
        y_pos_list = y_pos_list,
        gen_model2 = gen_model2
    )
    

    return ppo_agent, train_venv, valid_venv
    

if __name__ == '__main__':
    configs = parse_args()
    
    if configs.gpu >= 0:
        device = torch.device("cuda:{}".format(configs.gpu))
    else:
        device = torch.device("cpu")
    gen_model = Generator()
    gen_model.to(device)
    loadstr = "base1/" + "single_" + configs.env_name + '_' + str(0.05)
    gen_model.load_state_dict(torch.load(loadstr)) 
    
    mask_model = Generator()
    mask_model.to(device)
    loadstr = "base/" + "single_" + configs.env_name + '_' + str(0.05)
    mask_model.load_state_dict(torch.load(loadstr)) 
    mask_model.eval()
    
    openstr = 'patchs/' + configs.env_name + '_' + str(configs.percent_image)
    with open(openstr,'rb') as f:
    	patch = pkl.load(f)
    openstr = 'x_pos/' + configs.env_name + '_' + str(configs.percent_image)
    with open(openstr,'rb') as f:
    	x_pos = pkl.load(f)
    openstr = 'y_pos/' + configs.env_name + '_' + str(configs.percent_image)
    with open(openstr,'rb') as f:
    	y_pos = pkl.load(f)
    	
    
    ppo_agent, train_venv, valid_venv = env_set(configs)
    patch_size = patch.shape[0]
    replay_buffer = generate_buffer(configs, ppo_agent, train_venv, valid_venv)
    x_pos_list, y_pos_list = gradient_based_search(ppo_agent, 64, replay_buffer.sample(configs.roa_sample), patch_size, patch_size, int(patch_size/2), int(patch_size/2), configs.roa_region)
    print(x_pos_list)
    print(y_pos_list)
    ppo_agent, train_venv, valid_venv = env_set2(configs, patch_size, x_pos_list, y_pos_list, mask_model)
    
    ppo_agent, train_reward, train_std = gau_train(configs, ppo_agent, train_venv, valid_venv, patch_size, x_pos_list, y_pos_list, mask_model)
    #train_reward, train_std = gau_train_test(configs, ppo_agent, train_venv, valid_venv, per_std)
    
    #train_reward, train_std = train_distri(configs, ppo_agent, train_venv, patch, x_pos, y_pos, gen_model)
    
    
    model_save_path = 'trained_models/' + configs.env_name + '_' + str(configs.percent_image)
    ppo_agent.model.save_to_file(model_save_path)
    
   
