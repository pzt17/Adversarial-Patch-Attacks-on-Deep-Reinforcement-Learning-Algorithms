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
from PIL import Image as im
import cv2


parser = argparse.ArgumentParser(description = 'DQN')
parser.add_argument('--num_frames', type = int, default = 1100, metavar = 'NF', help = 'total number of frames')
parser.add_argument('--env', default =  "FreewayNoFrameskip-v4", metavar = 'ENV', help = 'environment to play on')
parser.add_argument('--load_model_path', default = "Freeway.pt", metavar = 'LMP',help = 'name of pth file of model')
parser.add_argument('--base_path', default = "checkpoint", metavar = 'BP', help = 'folder for trained models')
parser.add_argument('--env-config', default='config.json', metavar='EC', help='environment to1crop and resize info (default: config.json)')
parser.add_argument('--gpu-id', type=int, default=0,  help='GPUs to use [-1 CPU only] (default: -1)')
parser.add_argument('--test_frames', type = int, default = 10000)
parser.add_argument('--percent_image', type = float, default = 0.05)
parser.add_argument('--margin', type = int, default = 0)
parser.add_argument("--percent_rec", type = float, default = 0.05)

def patch_states(ori_states, patch, pos_x, pos_y):
	states = ori_states.clone()
	patch_len = patch.shape[2]
	states[0, 0, pos_y: pos_y + patch_len, pos_x: pos_x + patch_len] = patch[0, :, :] *(1.0/255.0)
	return states

def mask_state(ori_state, gen_state, patch_len, x_pos, y_pos):

	ori_state[:, :, x_pos:x_pos + patch_len, y_pos:y_pos + patch_len] = 1.0
	return ori_state
	
def patch_pos(ori_state, gen_state, patch_len):
	rec_side_num = 84 - patch_len + 1
	rec_sum = torch.zeros(rec_side_num * rec_side_num)
	diff = torch.abs(ori_state - gen_state)
	for i in range(rec_side_num):
		for j in range(rec_side_num):
			rec_sum[i*rec_side_num + j] = torch.sum(diff[0, 0, i:(i+patch_len), j: (j + patch_len)])
	_, mx_idx = torch.max(rec_sum, dim = 0)
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
			
	env = make_atari(args.env)
	env = wrap_deepmind(env, central_crop=True, clip_rewards=False, episode_life=False, **env_conf)
	env = wrap_pytorch(env)
	
	model = AMP(env.observation_space.shape[0], env.action_space)
	complete_model_path =  "mask_amp/" + args.env + '_' + str(args.percent_image)
	model.load_state_dict(torch.load(complete_model_path))
	with torch.cuda.device(args.gpu_id):
		model.cuda()
	model.eval()

	num_frames = args.num_frames
	percent_image = args.percent_image
	
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	
	gen_model = Generator()
	gen_model.to(device)
	loadstr = "base/" + "single_" + args.load_model_path + '_' + str(args.percent_rec)
	gen_model.load_state_dict(torch.load(loadstr)) 

	openstr = 'patchs/' + args.env + '_' + str(args.percent_image)
	with open(openstr,'rb') as f:
		patch = pkl.load(f)
	patch_len = patch.shape[1]
	patch_size =  int((1.0*84*84*args.percent_image)**0.5)
	#patch = np.zeros((1, patch_size, patch_size))
	print(patch)
	patch = torch.tensor(patch, requires_grad = False)

	
	openstr = 'x_pos/' + args.env + '_' + str(args.percent_image)
	with open(openstr,'rb') as f:
		x_pos = pkl.load(f)
	
	openstr = 'y_pos/' + args.env + '_' + str(args.percent_image)
	with open(openstr,'rb') as f:
		y_pos = pkl.load(f)
	
	
	print(x_pos)
	print(y_pos)
	state = env.reset()
	episode_reward = 0.0
	episode_record = []
	episode_cnt = 0.0

	for frames in range(num_frames):
		state = torch.tensor(np.float32(state), requires_grad = False).unsqueeze(0).to(device)
		
		if frames == 1000:
			state_img = state.squeeze(0).squeeze(0).detach().cpu().numpy() *255.0
			image_data = im.fromarray(state_img)
			if image_data.mode != 'RGB':
				image_data = image_data.convert('RGB')
			add_sv = args.env + '_origin.png'
			image_data.save(add_sv)
			
			
		#print(state[0, 0, 30:35, 30:35])
		patch_state = patch_states(state, patch, x_pos, y_pos)	
		
		if frames == 1000:
			state_img = patch_state.squeeze(0).squeeze(0).detach().cpu().numpy() *255.0
			image_data = im.fromarray(state_img)
			if image_data.mode != 'RGB':
				image_data = image_data.convert('RGB')
			add_sv = args.env + '_patch_state.png'
			image_data.save(add_sv)
			
		state_gen = gen_model(patch_state)
		
		if frames == 1000:
			state_img = state_gen.squeeze(0).squeeze(0).detach().cpu().numpy() *255.0
			image_data = im.fromarray(state_img)
			if image_data.mode != 'RGB':
				image_data = image_data.convert('RGB')
			add_sv = args.env + '_reconstruct.png'
			image_data.save(add_sv)
		
		if frames % 1000 == 0:
			x_pos_pred, y_pos_pred = patch_pos(patch_state, state_gen, patch_size + args.margin)
			print('x_pos:', x_pos_pred)
			print('y_pos:', y_pos_pred)
			
		state = mask_state(patch_state, state_gen, patch_size + args.margin, x_pos_pred, y_pos_pred)
		
		
		
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

	file1 = open("normal_reward.txt", "a")
	str_txt = "\n" + args.env + " per:" + str(args.percent_image)+ " ave:" + str(reward_mean) + " std:" + str(reward_std)
	file1.write(str_txt)
	file1.close()


	
	
