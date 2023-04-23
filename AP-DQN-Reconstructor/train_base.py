import numpy as np
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F
import numpy as np
from numpy import random
from torch.autograd import Variable
from collections import deque
import pickle as pkl
from generator import Generator
from qlay import state_lay
from qlay import Qlay
from qlay import mlay
from utils import read_config
from environment import make_atari, wrap_deepmind, wrap_pytorch
from dqn import DuelingCnnDQN


parser = argparse.ArgumentParser(description = 'DQN')
parser.add_argument('--num_frames', type = int, default = 10000, metavar = 'NF', help = 'total number of frames')
parser.add_argument('--env', default =  "FreewayNoFrameskip-v4", metavar = 'ENV', help = 'environment to play on')
parser.add_argument('--load_model_path', default = "Freeway.pt", metavar = 'LMP',help = 'name of pth file of model')
parser.add_argument('--base_path', default = "checkpoint", metavar = 'BP', help = 'folder for trained models')
parser.add_argument('--relay_capacity', type = int, default = 150000, metavar = 'RC', help = 'capacity of the relay')
parser.add_argument('--env-config', default='config.json', metavar='EC', help='environment to crop and resize info (default: config.json)')
parser.add_argument('--gpu-id', type=int, default=0,  help='GPUs to use [-1 CPU only] (default: -1)')
parser.add_argument("--lr", type=float, default=0.01, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--batch_size", type = int, default = 10)
parser.add_argument("--percent_image", type = float, default = 0.05)
			
def mstate_patch(images, patch_size):
	batch_size = images.shape[0]
	x_pos = np.random.randint(0, 84 - patch_size + 1, patch_size)
	y_pos = np.random.randint(0, 84 - patch_size + 1, patch_size)
	for i in range(batch_size):
		images[i, 0, x_pos[i]:x_pos[i] + patch_size, y_pos[i]: y_pos[i] + patch_size] = random.random()
	return images
	
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
		
	model = DuelingCnnDQN(env.observation_space.shape[0], env.action_space)
	complete_model_path = args.base_path + "/" + args.load_model_path
	weights = torch.load(complete_model_path, map_location=torch.device('cuda:{}'.format(args.gpu_id)))
	if "model_state_dict" in weights.keys():
		weights = weights['model_state_dict']
	model.load_state_dict(weights)
	
	with torch.cuda.device(args.gpu_id):
		model.cuda()
	model.eval()
	
	patch_len = int((1.0*84*84*args.percent_image)**0.5)
	gen_model = Generator()
	gen_model.to(device)
	optimizer = torch.optim.Adam(gen_model.parameters(), lr=args.lr)
	ploss = nn.MSELoss()
					
	state = env.reset()
	state_relay = state_lay(args.relay_capacity)
	tot_loss = 0.0
	
	for frame in range(args.num_frames):
		state_relay.push(state)
		state = torch.FloatTensor(state).unsqueeze(0).to(device)
		action = model.act(state)
		next_state, reward, done, _ = env.step(action)
		env.render()	
		state = next_state

		if done:
        		state = env.reset()
	
		if frame >= args.batch_size:			
			optimizer.zero_grad()
			states = state_relay.sample(args.batch_size)
			origin_states = states.copy()
			states = mstate_patch(states, patch_len)
			states = torch.tensor(np.float32(states), requires_grad = True).to(device)
			out_states = gen_model(states)
			origin_states = torch.tensor(np.float32(origin_states), requires_grad = False).to(device)
			loss = ploss(origin_states, out_states) 
			tot_loss = tot_loss + loss
			loss.backward()
			optimizer.step()
			
			if frame%500 == 0:
				print('loss:', tot_loss/500.0)
				tot_loss = 0.0
	
	pt_name = "base/" + "single_" + args.load_model_path + "_" + str(args.percent_image)
	torch.save(gen_model.state_dict(), pt_name)
	
	
