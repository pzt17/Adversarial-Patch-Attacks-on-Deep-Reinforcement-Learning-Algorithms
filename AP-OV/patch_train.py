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
from square import init_patch_square
from utils import read_config
from qlay import state_lay
from patch_gen import patch_gen
import pickle as pkl
from patch_test_mstd import patch_test_mstd
from patch_valid import patch_valid

parser = argparse.ArgumentParser(description = 'DQN')
parser.add_argument('--num_frames', type = int, default = 1000, metavar = 'NF', help = 'total number of frames')
#parser.add_argument('--env', default =  "FreewayNoFrameskip-v4", metavar = 'ENV', help = 'environment to play on')
parser.add_argument('--env', default = "Freeway", metavar = 'LMP',help = 'name of pth file of model')
parser.add_argument('--base_path', default = "checkpoint", metavar = 'BP', help = 'folder for trained models')
parser.add_argument('--xskip',type = int, default = 1, metavar = 'XS', help = 'skip size for x when searching for optimal occlusion rectangle')
parser.add_argument('--yskip',type = int, default = 1, metavar = 'YS', help = 'skip size for y when searching for optimal occlusion rectangle')
parser.add_argument('--num_rec', type = int, default = 5, metavar = 'NR', help = 'number of area that we put a grey rectangle on')
parser.add_argument('--pgd_iters', type = int, default = 500)
parser.add_argument('--alpha', type = float, default = 0.01)
parser.add_argument('--percent_image', type = float, default = 0.01, metavar = 'PI', help = 'patch is what percentage of the original image')
parser.add_argument('--relay_batch', type = int, default = 10, metavar = 'RB', help = 'batch size for traning patch')
parser.add_argument('--relay_sample', type = int, default = 128)
parser.add_argument('--relay_capacity', type = int, default = 100000, metavar = 'RB', help = 'batch size for traning patch')
parser.add_argument('--number_patch', type = int, default = 3, help = 'number of patchs')
parser.add_argument('--env-config', default='config.json', metavar='EC', help='environment to1crop and resize info (default: config.json)')
parser.add_argument('--gpu-id', type=int, default=0,  help='GPUs to use [-1 CPU only] (default: -1)')
parser.add_argument('--test_patch', type= bool, default= True)
parser.add_argument('--test_size', type = int, default = 512, help = 'number of patchs')
parser.add_argument('--test_frames', type = int, default = 10000)
parser.add_argument('--valid_frames', type = int, default = 10000)
parser.add_argument('--total_patch', type = int, default = 10, help = 'total of patchs')

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
	
	model = DuelingCnnDQN(env.observation_space.shape[0], env.action_space)
	#complete_model_path = args.base_path + "/" + args.load_model_path 
	complete_model_path = args.base_path + "/" + args.env + '.pt'
	weights = torch.load(complete_model_path, map_location=torch.device('cuda:{}'.format(args.gpu_id)))
	if "model_state_dict" in weights.keys():
		weights = weights['model_state_dict']
	model.load_state_dict(weights)
	
	with torch.cuda.device(args.gpu_id):
		model.cuda()
	model.eval()
	
	rec_width =  int((1.0*84*84*args.percent_image)**0.5)
	rec_height =  int((1.0*84*84*args.percent_image)**0.5)
	xskip = int(rec_width/2)
	yskip = int(rec_width/2)
	num_rec = args.num_rec
	num_frames = args.num_frames
	percent_image = args.percent_image
	relay_batch = args.relay_batch


	relay = state_lay(args.relay_capacity)
	openstr = 'states/' + env_name + '.pkl'
	with open(openstr,'rb') as f:
		relay = pkl.load(f)
	
	mean_rewards = np.zeros(args.total_patch)
	for patch_cnt in range(args.total_patch):
		state = env.reset()
		optimal_patch = init_patch_square(state.shape[2], percent_image)
		optimal_reward = 1000000000
		attack_x = 1
		attack_y = 1
		for patch_iter in range(args.number_patch):
			patch = init_patch_square(state.shape[2], percent_image)
			patch, ax, ay = patch_gen(patch, model, env, relay, args.relay_sample, state.shape[2], rec_width, rec_height, xskip, yskip, num_rec, args.relay_capacity, relay_batch, num_frames)
			cur_reward = patch_valid(patch, ax, ay, model, env, args.valid_frames)
			print('reward:', cur_reward)
			if cur_reward < optimal_reward:
				optimal_reward = cur_reward
				attack_x = ax
				attack_y = ay
				optimal_patch = patch
	
		reward_mean, reward_std = patch_test_mstd(optimal_patch, attack_x, attack_y, model, state.shape[1], env, args.test_frames)
		mean_rewards[patch_cnt] = reward_mean
		print('reward mean:', reward_mean)
		print('reward std:', reward_std)
		
		popen = 'patchs/' + args.env + '/' + str(args.percent_image) + '_' + str(patch_cnt)
		with open(popen, 'wb') as f:
			pkl.dump(optimal_patch, f)
	
		pattackx = 'x_pos/' + args.env  + "/" + str(args.percent_image) + '_' + str(patch_cnt)
		with open(pattackx, 'wb') as f1:
			pkl.dump(attack_x, f1)
	
		pattacky = 'y_pos/' + args.env + "/" + str(args.percent_image) + '_' + str(patch_cnt)
		with open(pattacky, 'wb') as f2:
			pkl.dump(attack_y, f2)
		
	mean_of_mean = format(np.mean(mean_rewards), ".2f")
	std_of_mean = format(np.std(mean_rewards), ".2f")
	
	file1 = open("reward.txt", "a")
	str_txt = "\n" + args.env + " percent image " + str(args.percent_image)+ ": " + str(mean_of_mean) + " \pm " + str(std_of_mean)
	file1.write(str_txt)
	file1.close()
	
