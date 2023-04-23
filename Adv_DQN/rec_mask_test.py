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
from generator import Generator
import copy

parser = argparse.ArgumentParser(description = 'DQN')
parser.add_argument('--num_frames', type = int, default = 10000, metavar = 'NF', help = 'total number of frames')
parser.add_argument('--env', default =  "Freeway", metavar = 'ENV', help = 'environment to play on')
parser.add_argument('--base_path', default = "checkpoint", metavar = 'BP', help = 'folder for trained models')
parser.add_argument('--env-config', default='config.json', metavar='EC', help='environment to1crop and resize info (default: config.json)')
parser.add_argument('--gpu-id', type=int, default=0,  help='GPUs to use [-1 CPU only] (default: -1)')
parser.add_argument('--test_frames', type = int, default = 10000)
parser.add_argument('--percent_image', type = float, default = 0.05)
parser.add_argument('--margin', type = int, default = 0)
parser.add_argument("--percent_rec", type = float, default = 0.05)

def patch_states(states, patch, pos_x, pos_y):
	patch_len = patch.shape[2]
	states[0, 0, pos_y: pos_y + patch_len, pos_x: pos_x + patch_len] = patch[0, :, :] *(1.0/255.0)
	return states

def mask_state(ori_state, patch_len, x_pos, y_pos):

	ori_state[:, :, x_pos:x_pos + patch_len, y_pos:y_pos + patch_len] = 1.0
	return ori_state
	
def syn_state(ori_state, patch_len, x_pos, y_pos):
        ori_state[0, 0, x_pos: x_pos + patch_len, y_pos: y_pos + patch_len] = 1.5
        return ori_state
	
def patch_pos(ori_state, gen_state, patch_len):
	rec_side_num = 84 - patch_len + 1
	rec_sum = torch.zeros(rec_side_num * rec_side_num)
	m = nn.AvgPool2d(kernel_size = patch_len, stride = 1)
	m2 = nn.MaxPool2d(kernel_size = rec_side_num, stride = 1, return_indices = True)
	diff = torch.abs(ori_state - gen_state).squeeze(0)

	mx_val = m(diff)
	_, mx_idx = m2(mx_val)
	mx_idx = mx_idx.squeeze(0).squeeze(0)
			
	x_pos = mx_idx // rec_side_num
	y_pos = mx_idx % rec_side_num
	
	return x_pos, y_pos

if __name__ == '__main__':
	args = parser.parse_args()
	num_frames = args.num_frames
	
	torch.cuda.empty_cache()
	
	setup_json = read_config(args.env_config)
	env_conf = setup_json["Default"]
	for i in setup_json.keys():
		if i in args.env:
			env_conf = setup_json[i]
			
	
	env_name = args.env + 'NoFrameskip-v4'		
	env = make_atari(env_name)
	env = wrap_deepmind(env, central_crop=True, clip_rewards=False, episode_life=False, **env_conf)
	env = wrap_pytorch(env)
	
	model = AMP(env.observation_space.shape[0], env.action_space)
	env_name = args.env + 'NoFrameskip-v4'
	complete_model_path =  "mask_amp/" + env_name + '_' + str(args.percent_image)
	model.load_state_dict(torch.load(complete_model_path))
	with torch.cuda.device(args.gpu_id):
		model.cuda()
	model.eval()

	num_frames = args.num_frames
	percent_image = args.percent_image
	
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	
	gen_model = Generator()
	gen_model.to(device)
	load_model_path = args.env + '.pt'
	loadstr = "base/" + "single_" + load_model_path + '_' + str(args.percent_rec)
	gen_model.load_state_dict(torch.load(loadstr)) 
	gen_model.eval()

	gen_model2 = Generator()
	gen_model2.to(device)
	load_model_path = args.env + '.pt'
	loadstr = "base2/" + "single_" + load_model_path + '_' + str(args.percent_rec)
	gen_model2.load_state_dict(torch.load(loadstr)) 
	gen_model2.eval()

	episode_record = []
	for patch_cnt in range(10):
		print('patch cnt:', patch_cnt)
		print('percent:', args.percent_image)
		openstr = 'patchs/' + args.env + '/' + str(args.percent_image) + '_' + str(patch_cnt)
		with open(openstr,'rb') as f:
			patch = pkl.load(f)
		patch_len = patch.shape[1]
		patch_size =  int((1.0*84*84*args.percent_image)**0.5)
		print(patch)
		patch = torch.tensor(patch, requires_grad = False)

	
		openstr = 'x_pos/' + args.env + '/' + str(args.percent_image) + '_' + str(patch_cnt)
		with open(openstr,'rb') as f:
			x_pos = pkl.load(f)
	
		openstr = 'y_pos/' + args.env + '/' + str(args.percent_image) + '_' + str(patch_cnt)
		with open(openstr,'rb') as f:
			y_pos = pkl.load(f)
			
		state = env.reset()
		episode_reward = 0.0
		print('x_pos:', x_pos)
		print('y_pos:', y_pos)
		for frames in range(num_frames):
			state = torch.tensor(np.float32(state), requires_grad = False).unsqueeze(0).to(device)
			patch_state = patch_states(state, patch, x_pos, y_pos)	
			state_gen = gen_model(patch_state)
		
			x_pos_pred, y_pos_pred = patch_pos(patch_state, state_gen, patch_size)
			
			state = mask_state(patch_state, patch_size, x_pos_pred, y_pos_pred)
			state = gen_model2(state)
			state2 = syn_state(patch_state, patch_size, x_pos_pred, y_pos_pred)
			state = torch.where(state2 < 1.4, state2, state)
		
			action = model.act(state)
			next_state, reward, done, _ = env.step(action)			
			env.render()
			state = next_state
			episode_reward += reward
		
			if done:
				state = env.reset()
				print('Episode Reward in Testing Phase:', episode_reward)
				episode_record.append(episode_reward)
				episode_reward = 0.0
	

	reward_mean =  format(np.mean(episode_record),".2f")
	reward_std = format(np.std(episode_record), ".2f")
	print('reward mean:', reward_mean)
	print('reward std:', reward_std)

	file1 = open("reward.txt", "a")
	str_txt = "\n" + args.env + " per:" + str(args.percent_image)+ " ave:" + str(reward_mean) + " std:" + str(reward_std)
	file1.write(str_txt)
	file1.close()


	
	
