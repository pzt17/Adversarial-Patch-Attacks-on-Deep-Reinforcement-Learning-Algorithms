import math, random
import gym
import numpy as np
import csv
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
from qlay import state_lay
import pickle as pkl


parser = argparse.ArgumentParser(description = 'DQN')
parser.add_argument('--num_frames', type = int, default = 20000, metavar = 'NF', help = 'total number of frames')
parser.add_argument('--env', default =  "FreewayNoFrameskip-v4", metavar = 'ENV', help = 'environment to play on')
parser.add_argument('--load_model_path', default = "Freeway.pt", metavar = 'LMP',help = 'name of pth file of model')
parser.add_argument('--base_path', default = "checkpoint", metavar = 'BP', help = 'folder for trained models')
parser.add_argument('--env-config', default='config.json', metavar='EC', help='environment to1crop and resize info (default: config.json)')
parser.add_argument('--gpu-id', type=int, default=0,  help='GPUs to use [-1 CPU only] (default: -1)')
parser.add_argument('--test_frames', type = int, default = 10000)
parser.add_argument('--percent_image', type = float, default = 0.05)
parser.add_argument('--dec', type = int, default = 10)

def patch_states(states, patch, pos_x, pos_y):
	patch_size = patch.shape[2]
	states[0,  pos_y: pos_y + patch_size, pos_x: pos_x + patch_size] = 0.0
	return states
	
def mstate_patch(states, region_size):
	x_pos = np.random.randint(0, 84 - region_size + 1)
	y_pos = np.random.randint(0, 84 - region_size + 1)
	for j in range(region_size):
		for k in range(region_size):
			states[:, :, x_pos + j, y_pos + k] = 0.0
	return states

if __name__ == '__main__':
	args = parser.parse_args()
	num_frames = args.num_frames
	
	torch.cuda.empty_cache()
	
	setup_json = read_config(args.env_config)
	env_conf = setup_json["Default"]
	for i in setup_json.keys():
		if i in args.env:
			env_conf = setup_json[i]
			
	env = make_atari(args.env)
	env = wrap_deepmind(env, central_crop=True, clip_rewards=False, episode_life=False, **env_conf)
	env = wrap_pytorch(env)
	
	model = DuelingCnnDQN(env.observation_space.shape[0], env.action_space)
	complete_model_path =  "mask_amp/" + args.env + '_' + str(args.percent_image)
	model.load_state_dict(torch.load(complete_model_path))
	with torch.cuda.device(args.gpu_id):
		model.cuda()
	model.eval()

	num_frames = args.num_frames
	percent_image = args.percent_image
	
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	

	openstr = 'patchs/' + args.env + '_' + str(args.percent_image)
	with open(openstr,'rb') as f:
		patch = pkl.load(f)
	patch_len = patch.shape[1]
	patch_size =  int((1.0*84*84*args.percent_image)**0.5)

	
	openstr = 'x_pos/' + args.env + '_' + str(args.percent_image)
	with open(openstr,'rb') as f:
		x_pos = pkl.load(f)
	
	openstr = 'y_pos/' + args.env + '_' + str(args.percent_image)
	with open(openstr,'rb') as f:
		y_pos = pkl.load(f)
	
	state = env.reset()
	episode_reward = 0.0
	episode_record = []
	episode_cnt = 0.0
	softm = nn.Softmax(dim = 0)
	q_diff_list = []
	
	for frames in range(num_frames):
		state = patch_states(state, patch, x_pos, y_pos)
		state = torch.tensor(np.float32(state)).unsqueeze(0).to(device)
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
	
	reward_mean = 0.0
	reward_std = 0.0
	if episode_cnt == 0:
		reward_mean =  0.0
		reward_std =  0.0
	else:
		reward_mean =  np.mean(episode_record)
		reward_std = np.std(episode_record)
	
	
	print('reward mean:', reward_mean)
	print('reward std:', reward_std)

	file1 = open("known_random_reward.txt", "a")
	str_txt = "\n" + args.env + " per:" + str(args.percent_image)+ " ave:" + str(reward_mean) + " std:" + str(reward_std)
	file1.write(str_txt)
	file1.close()


	
	
