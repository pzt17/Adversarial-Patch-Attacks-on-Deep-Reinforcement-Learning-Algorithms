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
from patch_test_episode import patch_test_episode
from patch_valid import patch_valid

parser = argparse.ArgumentParser(description = 'DQN')
parser.add_argument('--num_frames', type = int, default = 1000, metavar = 'NF', help = 'total number of frames')
#parser.add_argument('--env', default =  "FreewayNoFrameskip-v4", metavar = 'ENV', help = 'environment to play on')
parser.add_argument('--env', default = "Freeway", metavar = 'LMP',help = 'name of pth file of model')
parser.add_argument('--base_path', default = "checkpoint", metavar = 'BP', help = 'folder for trained models')
parser.add_argument('--xskip',type = int, default = 5, metavar = 'XS', help = 'skip size for x when searching for optimal occlusion rectangle')
parser.add_argument('--yskip',type = int, default = 5, metavar = 'YS', help = 'skip size for y when searching for optimal occlusion rectangle')
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
	xskip = args.xskip
	yskip = args.yskip
	num_rec = args.num_rec
	num_frames = args.num_frames
	percent_image = args.percent_image
	relay_batch = args.relay_batch


	relay = state_lay(args.relay_capacity)
	openstr = 'states/' + env_name + '.pkl'
	with open(openstr,'rb') as f:
		relay = pkl.load(f)
	
	episode_rewards = []
	for patch_cnt in range(args.total_patch):
		print('patch_cnt:', patch_cnt)
		
		openstr = 'patchs/' + args.env + '/' + str(args.percent_image) + '_' + str(patch_cnt)
		with open(openstr,'rb') as f:
			patch = pkl.load(f)
		patch_len = patch.shape[1]
		patch_size =  int((1.0*84*84*args.percent_image)**0.5)

		openstr = 'x_pos/' + args.env + '/' + str(args.percent_image) + '_' + str(patch_cnt)
		with open(openstr,'rb') as f:
			x_pos = pkl.load(f)
	
		openstr = 'y_pos/' + args.env + '/' + str(args.percent_image) + '_' + str(patch_cnt)
		with open(openstr,'rb') as f:
			y_pos = pkl.load(f)
	
		current_episode_reward = patch_test_episode(patch, x_pos, y_pos, model, 84, env, args.test_frames)
		episode_rewards.extend(current_episode_reward)
		
		
		
	mean_of_mean = format(np.mean(episode_rewards), ".2f")
	std_of_mean = format(np.std(episode_rewards), ".2f")
	
	file1 = open("std_reward.txt", "a")
	str_txt = "\n" + args.env + " percent image " + str(args.percent_image)+ ": " + str(mean_of_mean) + " \pm " + str(std_of_mean)
	file1.write(str_txt)
	file1.close()
	
