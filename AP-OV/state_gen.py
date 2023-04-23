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
from add_relay import add_relay


parser = argparse.ArgumentParser(description = 'DQN')
parser.add_argument('--num_frames', type = int, default = 10000, metavar = 'NF', help = 'total number of frames')
parser.add_argument('--env', default =  "FreewayNoFrameskip-v4", metavar = 'ENV', help = 'environment to play on')
parser.add_argument('--load_model_path', default = "Freeway.pt", metavar = 'LMP',help = 'name of pth file of model')
parser.add_argument('--base_path', default = "checkpoint", metavar = 'BP', help = 'folder for trained models')
parser.add_argument('--relay_capacity', type = int, default = 150000, metavar = 'RC', help = 'capacity of the relay')
parser.add_argument('--env-config', default='config.json', metavar='EC', help='environment to crop and resize info (default: config.json)')
parser.add_argument('--gpu-id', type=int, default=0,  help='GPUs to use [-1 CPU only] (default: -1)')

if __name__ == '__main__':
	args = parser.parse_args()
	
	setup_json = read_config(args.env_config)
	env_conf = setup_json["Default"]
	for i in setup_json.keys():
		if i in args.env:
			env_conf = setup_json[i]
			
	env = make_atari(args.env)
	env = wrap_deepmind(env, central_crop=True, clip_rewards=False, episode_life=False, **env_conf)
	env = wrap_pytorch(env)
	
	model = DuelingCnnDQN(env.observation_space.shape[0], env.action_space)
	complete_model_path = args.base_path + "/" + args.load_model_path
	weights = torch.load(complete_model_path, map_location=torch.device('cuda:{}'.format(args.gpu_id)))
	if "model_state_dict" in weights.keys():
		weights = weights['model_state_dict']
	model.load_state_dict(weights)
	
	with torch.cuda.device(args.gpu_id):
		model.cuda()
	model.eval()
	
	num_frames = args.num_frames
	relay_capacity = args.relay_capacity
	
	state_relay = state_lay(relay_capacity)
	
	#for per_correct in range(0,105,10):
		#print('percent_correct:', per_correct)
	state_relay = add_relay(state_relay, num_frames, env, 0.0/100.0, model)
	state_relay = add_relay(state_relay, num_frames, env, 35.0/100.0, model)
	
	openstr = 'states/' + args.env + '.pkl'
	with open(openstr, 'wb') as f:
		pkl.dump(state_relay, f)
	
	
