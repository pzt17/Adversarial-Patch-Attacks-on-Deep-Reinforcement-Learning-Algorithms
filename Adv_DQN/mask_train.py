import math, random
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F
import copy
from IPython.display import clear_output
import matplotlib.pyplot as plt
from environment import make_atari, wrap_deepmind, wrap_pytorch
import argparse
import pickle as pkl
from dqn import DuelingCnnDQN
from dqn import AMP
from utils import read_config
from qlay import Qlay
from qlay import Qlay2
from patch_pos import gradient_based_search
from generator import Generator

parser = argparse.ArgumentParser(description = 'DQN')
parser.add_argument('--num_frames', type = int, default = 100000, metavar = 'NF', help = 'total number of frames')
parser.add_argument('--env', default =  "FreewayNoFrameskip-v4", metavar = 'ENV', help = 'environment to play on')
parser.add_argument('--load_model_path', default = "Freeway.pt", metavar = 'LMP',help = 'name of pth file of model')
parser.add_argument('--base_path', default = "checkpoint", metavar = 'BP', help = 'folder for trained models')
parser.add_argument('--relay_sample', type = int, default = 4)
parser.add_argument('--relay_capacity', type = int, default = 10000, metavar = 'RB', help = 'batch size for traning patch')
parser.add_argument('--env-config', default='config.json', metavar='EC', help='environment to1crop and resize info (default: config.json)')
parser.add_argument('--gpu-id', type=int, default=0,  help='GPUs to use [-1 CPU only] (default: -1)')
parser.add_argument('--lr', type = float, default = 0.000125)
parser.add_argument('--tem_constant', type = int, default = 1)
parser.add_argument('--kappa', type = float, default = 0.2)
parser.add_argument('--continue_train', type = int, default = 0)
parser.add_argument('--percent_image', type = float, default = 0.02)
parser.add_argument('--roa_frames', type = int, default = 10000)
parser.add_argument('--roa_sample', type = int, default = 512)
parser.add_argument('--roa_region', type = int, default = 20)
parser.add_argument('--c_constant', type = float, default = 0.05)
parser.add_argument('--pgd_step', type = float, default = 0.07)
parser.add_argument('--beta', type = float, default = 0.1)
parser.add_argument('--buffer_size', type = int, default = 5000)
parser.add_argument('--batch_size', type = int, default = 32)
parser.add_argument('--gamma', type = float, default = 0.99)
parser.add_argument('--epsilon', type = float, default = 0.0)


args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
gen_model = Generator()
gen_model.to(device)
loadstr = "base2/" + "single_" + args.load_model_path + '_' + str(0.05)
gen_model.load_state_dict(torch.load(loadstr)) 

class ReplayBuffer(object):
    def __init__(self, capacity, device):
        self.capacity = capacity
        self.device = device
        self.buffer = []
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (torch.cat(state, dim=0), torch.cat(action, dim=0), torch.cat(reward, dim=0), 
                torch.cat(next_state, dim =0), torch.cat(done, dim=0))
        
    def __len__(self):
        return len(self.buffer)


        
def _compute_loss(curr_model, target_model, data, gamma, device):
    state, action, reward, next_state, done = data
    
    q_values = curr_model(state)
    next_q_values = curr_model(next_state)
    target_next_q_values = target_model(next_state)

    q_value       = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value = target_next_q_values.gather(1, torch.argmax(next_q_values, 1, keepdim=True)).squeeze(1)
    expected_q_value = reward + gamma * next_q_value * (1 - done)
    
    standard_loss = torch.min((q_value - expected_q_value.detach()).pow(2), torch.abs(q_value - expected_q_value.detach()))
    standard_loss = standard_loss.mean()
    
    return standard_loss

def mstate_patch(states_ori, region_size, y_pos_list, x_pos_list, model):
	batch_size = states_ori.shape[0]
	states = copy.deepcopy(states_ori)
	half_size = int(batch_size/2)
	
	x_pos = np.random.randint(0, 84 - region_size + 1, half_size)
	y_pos = np.random.randint(0, 84 - region_size + 1, half_size)
	for i in range(half_size):
		states[i, 0, x_pos[i]: x_pos[i] + region_size, y_pos[i]:y_pos[i] + region_size] = 1.0
	
	
	list_size = y_pos_list.shape[0]
	y_r = np.random.randint(0, list_size, half_size)
	x_r = np.random.randint(0, list_size, half_size)
	
	y_pos2 = y_pos_list[y_r]
	x_pos2 = x_pos_list[x_r]
	for i in range(half_size):
		states[half_size + i, 0, x_pos2[i]: x_pos2[i] + region_size, y_pos2[i]:y_pos2[i] + region_size] = 1.0
			
	states = torch.tensor(np.float32(states)).to(device)
	gen_state = gen_model(states)
	states = states.detach().cpu().numpy()
	gen_state = gen_state.detach().cpu().numpy()
	
	for i in range(half_size):
		states[i, 0, x_pos[i]: x_pos[i] + region_size, y_pos[i]:y_pos[i] + region_size] = gen_state[i, 0, x_pos[i]: x_pos[i] + region_size, y_pos[i]:y_pos[i] + region_size]
		states[half_size + i, 0, x_pos2[i]: x_pos2[i] + region_size, y_pos2[i]:y_pos2[i] + region_size] = gen_state[half_size + i, 0, x_pos2[i]: x_pos2[i] + region_size, y_pos2[i]:y_pos2[i] + region_size]
	
	
	return states
	



if __name__ == '__main__':
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
	complete_model_path = args.base_path + "/" + args.load_model_path
	weights = torch.load(complete_model_path, map_location=torch.device('cuda:{}'.format(args.gpu_id)))
	if "model_state_dict" in weights.keys():
		weights = weights['model_state_dict']
	model.load_state_dict(weights)
	
	with torch.cuda.device(args.gpu_id):
		model.cuda()
	model.eval()
	
	current_model = AMP(env.observation_space.shape[0], env.action_space)	
	target_model = AMP(env.observation_space.shape[0], env.action_space)	
	current_model.to(device)
	target_model.to(device)
	current_model.load_state_dict(weights)
	target_model.load_state_dict(current_model.state_dict())

	closs = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(current_model.parameters(), lr = args.lr)
	state_buffer = Qlay2(args.relay_capacity)
	patch_size =  int((1.0*84*84*args.percent_image)**0.5)
	
	state = env.reset()
	for frame in range(args.roa_frames):
		state_buffer.push(state)
		state = torch.FloatTensor(state).unsqueeze(0).to(device)
		action = model.act(state)
		next_state, reward, done, _ = env.step(action)	
		env.render()
		state = next_state
		
		if done:
			state = env.reset()
			
	skip_size = int(patch_size/2)
	y_pos_list, x_pos_list = gradient_based_search(model, 84, state_buffer.sample(args.roa_sample), patch_size, patch_size, skip_size, skip_size, args.roa_region)
	
	print(y_pos_list)
	print(x_pos_list)
	
	m = nn.Softmax(dim = 1)
	
	
	replay_buffer = ReplayBuffer(args.buffer_size, device)
	state_buffer = Qlay(args.relay_capacity, env.action_space.n)
	state = env.reset()
	m = nn.Softmax(dim = 1)
	background_patch = np.zeros_like(state)
	background_patch[:,:,:] = 0.5
	for frame in range(args.num_frames):
		state = torch.FloatTensor(state).unsqueeze(0).to(device)
		action = model.act(state, epsilon = args.epsilon)
		state_buffer.push(state.squeeze(0).detach().cpu().numpy(), action.detach().cpu().numpy())
		next_state, reward, done, _ = env.step(action)	
		env.render()
		
		next_state_b = torch.FloatTensor(next_state).unsqueeze(0).to(device)
		action_b = torch.LongTensor([action]).to(device)
		reward_b = torch.clamp(torch.FloatTensor([reward]).to(device), min=-1, max=1)
		done_b =  torch.FloatTensor([done]).to(device)
		replay_buffer.push(state, action_b, reward_b, next_state_b, done_b)
		
		state = next_state
		
		
		if frame > 10000:
			data = replay_buffer.sample(args.batch_size)
			states_ten, _, _, _, _ = data
			states = states_ten.detach().cpu().numpy()
			masked_states = mstate_patch(states, patch_size, y_pos_list, x_pos_list, model)
			
			optimizer.zero_grad()
			states_grad = torch.tensor(np.float32(masked_states), requires_grad = True).to(device)
			pred_q_val = current_model(states_grad)
			states_nograd = torch.tensor(np.float32(states), requires_grad = False).to(device)
			real_q_val = model(states_nograd)
			mx_action_idx = torch.argmax(real_q_val, dim = 1)
			mx_action_idx = mx_action_idx.unsqueeze(1)
			mx_action_val = torch.gather(pred_q_val, 1, mx_action_idx) - args.c_constant
			
			diff = torch.max(pred_q_val - mx_action_val, torch.zeros_like(pred_q_val))
			advloss = diff.mean()
			stdloss = _compute_loss(current_model, target_model, data, args.gamma, device)
			
			loss = args.beta* stdloss + (1 - args.beta)*advloss
			loss.backward()
			optimizer.step()
			
			if frame%1000 == 0:
				target_model.load_state_dict(current_model.state_dict())
			
			if frame%1000 == 0:
				print(' ')
				print('percent:', args.percent_image)
				print('frame:', frame)
				print('loss:', loss)
				print('pred_q_val', pred_q_val[5])
				print('real_q_val', real_q_val[5])
				pred = torch.argmax(pred_q_val, dim = 1)
				real = torch.argmax(real_q_val, dim = 1)
				cor_cnt = 0
				for idx in range(args.batch_size):
					if pred[idx] == real[idx]:
						cor_cnt += 1
				cor_rate = 1.0*cor_cnt/(args.batch_size)
				print('pred_action:', torch.argmax(pred_q_val, dim = 1))
				print('real_action:', torch.argmax(real_q_val, dim = 1))
				print('accuracy:', cor_rate)
				print('diff loss:', args.beta* torch.sum(diff))
				
		
		if done:
			state = env.reset()

			
	pt_name = "mask_amp/" + args.env + '_' + str(args.percent_image)
	torch.save(current_model.state_dict(), pt_name)

	
