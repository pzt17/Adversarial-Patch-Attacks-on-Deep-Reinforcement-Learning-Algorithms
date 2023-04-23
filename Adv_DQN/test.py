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
import argparse
import pickle as pkl
from dqn import DuelingCnnDQN
from dqn import AMP
from utils import read_config
from qlay import Qlay
from dqn import AMP

parser = argparse.ArgumentParser(description = 'DQN')
parser.add_argument('--num_frames', type = int, default = 30000, metavar = 'NF', help = 'total number of frames')
parser.add_argument('--env', default =  "FreewayNoFrameskip-v4", metavar = 'ENV', help = 'environment to play on')
parser.add_argument('--load_model_path', default = "Freeway.pt", metavar = 'LMP',help = 'name of pth file of model')
parser.add_argument('--base_path', default = "checkpoint", metavar = 'BP', help = 'folder for trained models')
parser.add_argument('--env-config', default='config.json', metavar='EC', help='environment to crop and resize info (default: config.json)')
parser.add_argument('--gpu-id', type=int, default=0,  help='GPUs to use [-1 CPU only] (default: -1)')

	
if __name__ == "__main__":
	args = parser.parse_args()
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	
	setup_json = read_config(args.env_config)
	env_conf = setup_json["Default"]
	for i in setup_json.keys():
		if i in args.env:
			env_conf = setup_json[i]
			
	env = make_atari(args.env)
	env = wrap_deepmind(env, central_crop=True, clip_rewards=False, episode_life=False, **env_conf)
	env = wrap_pytorch(env)
		
	model = AMP(env.observation_space.shape[0], env.action_space)
	complete_model_path = "amp/" + args.env + '_'
	model.load_state_dict(torch.load(complete_model_path))
	
	with torch.cuda.device(args.gpu_id):
		model.cuda()
	model.eval()
	
	episode_reward = 0.0
	episode_record = []
	episode_cnt = 0.0
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	state = env.reset()
	
	for frames in range(args.num_frames):
		state = torch.FloatTensor(np.float32(state)).unsqueeze(0).to(device)
		action = model.act(state)
		next_state, reward, done, _ = env.step(action)
		env.render()
		state = next_state
		episode_reward += reward
		
		if done:
        		state = env.reset()
        		print('Episode Reward in Testing Phase:', episode_reward)
        		episode_record.append(episode_reward)
        		episode_cnt += 1
        		episode_reward = 0.0
        		
	if episode_cnt == 0:
		reward_mean =  0.0
		reward_std = 0.0
	else:
		reward_mean = np.mean(episode_record)
		reward_std =  np.std(episode_record)
		
	print(reward_mean)
	print(reward_std)
