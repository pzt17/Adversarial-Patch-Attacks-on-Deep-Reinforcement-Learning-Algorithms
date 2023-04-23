import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import torch.autograd as autograd 
from dqn import DuelingCnnDQN
from torch.autograd import Variable
import copy

def gradient_based_search(model, size, states, width, height, xskip, yskip, potential_nums):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        states_ori = copy.deepcopy(states)
        states = torch.tensor(states).to(device)
        states_grad = torch.tensor(states_ori).to(device)
        states_grad = Variable(states_grad, requires_grad = True)
        qvs = model.get_q_value(states)
        qvs = Variable(qvs, requires_grad = False)
        ms = nn.Softmax(dim = 1)
        attacked_q_values = ms(model.get_q_value(states_grad))
        loss = torch.sum(attacked_q_values * qvs)
        loss.backward()
        
        sum_grad = torch.sum(states_grad.grad, dim = 0)
        sum_grad = torch.sum(sum_grad, dim = 0)
        mx_val, _ = torch.max(torch.abs(sum_grad.view(sum_grad.shape[0]*sum_grad.shape[1], -1)), dim = 0)
        sum_grad = sum_grad / mx_val
        
        xcnt = ((size - width)//xskip) + 1
        ycnt = ((size - height)//yskip) + 1
        losses = torch.zeros(xcnt*ycnt)
        for i in range(xcnt):
        	for j in range(ycnt):
        		tm = sum_grad[i*xskip: i*xskip + width, j*yskip: j*yskip + height]
        		losses[i*xcnt + j] = torch.sum(torch.sum(torch.mul(tm,tm),1))
        tk_, tpk_idx = torch.topk(losses, potential_nums)
        xidx = tpk_idx // xcnt
        yidx = tpk_idx % xcnt
        
        opt_x = 1
        opt_y = 1
        mx_loss = 100000000
        with torch.set_grad_enabled(False):
        	for i in range(potential_nums):
        		patch_grey = states.clone()
        		patch_grey[:, :, xskip*xidx[i]:xskip*xidx[i] + width, yskip*yidx[i]:yskip*yidx[i] + height] = 0.5
        		cur_loss = torch.sum(ms(model.get_q_value(patch_grey))*qvs)
        		if cur_loss < mx_loss:
        			mx_loss = cur_loss
        			opt_x = xidx[i]*xskip
        			opt_y = yidx[i]*yskip
	
        return opt_x, opt_y
       
