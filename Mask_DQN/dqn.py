from __future__ import division
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

class DuelingCnnDQN(nn.Module):
    def __init__(self, num_channels, action_space):
        super(DuelingCnnDQN, self).__init__()
        self.num_actions = action_space.n
        print(self.num_actions)
        self.cnn = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.advantage = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )

        self.value = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        cnn = self.cnn(x)
        advantage = self.advantage(cnn)
        value = self.value(cnn)
        return value + advantage - torch.mean(advantage, dim=1, keepdim=True)
    
    
    def act(self, state, epsilon = 0.0):
        with torch.no_grad():
            if random.random() > epsilon:
                q_values = self.forward(state)
                action  = torch.argmax(q_values, dim=1)[0]
            else:
                action = torch.tensor(random.randrange(self.num_actions))
        return action
        
    def act_ran(self, state, ran_fac):
        with torch.no_grad():
            if random.random() > ran_fac:
                q_values = self.forward(state)
                action  = torch.argmax(q_values, dim=1)[0]
            else:
                action = torch.tensor(random.randrange(self.num_actions))
        return action
    
    def get_q_value(self, state):
        q_value= self.forward(state)
        return q_value 
        
    def get_q_value_np(self,state):
    	state   = state.unsqueeze(0)
    	q_value = self.forward(state)
    	return q_value
    	
    def act_mul(self, state):
        with torch.no_grad():
                q_values = self.forward(state)
                action  = torch.argmax(q_values, dim=1)
        return action
        
    def sfmx(self, state, tem_constant):
    	with torch.no_grad():
    		q_values = self.forward(state)
    		q_values = q_values * tem_constant
    		m = nn.Softmax(dim = 1)
    		res_x = m(q_values)
    	return res_x
    	
    def tem(self, state, tem_constant):
    	with torch.no_grad():
    		q_values = self.forward(state)
    		q_values = q_values * tem_constant
    	return q_values
    	
class AMP(nn.Module):
    def __init__(self, num_channels, action_space):
        super(AMP, self).__init__()
        self.num_actions = action_space.n
        print(self.num_actions)
        self.cnn = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.advantage = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )

        self.value = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        cnn = self.cnn(x)
        advantage = self.advantage(cnn)
        value = self.value(cnn)
        return value + advantage 
        
    def value_forward(self, x):
        cnn = self.cnn(x)
        value = self.value(cnn)
        return value

    def sfmx(self, state):
    	q_values = self.forward(state)
    	m = nn.Softmax(dim = 1)
    	res_x = m(q_values)
    	return res_x

    def act(self, state, epsilon = 0.0):
        with torch.no_grad():
            if random.random() > epsilon:
                prob = self.forward(state)
                action  = torch.argmax(prob, dim=1)[0]
            else:
                action = random.randrange(self.num_actions)
        return action

    def get_q_value(self, state):
        q_value= self.forward(state)
        return q_value 

	

	
	

class CnnDQN(nn.Module):
    def __init__(self, num_channels, action_space):
        super(CnnDQN, self).__init__()
        
        self.num_actions = action_space.n
        self.model = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(64*6*6, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )
        self.train()
        
    def forward(self, x):
        x = self.model(x)
        return x

    def act(self, state, epsilon = 0.0):
        with torch.no_grad():
            if random.random() > epsilon:
                q_value = self.forward(state)
                #print(q_value)
                action  = torch.argmax(q_value, dim=1)[0]
            else:
                action = random.randrange(self.num_actions)
        return action

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
    
class AutoLirpa_CnnDQN(nn.Module):
    def __init__(self, num_channels, action_space):
        super(AutoLirpa_CnnDQN, self).__init__()
        
        self.num_actions = action_space.n
        self.model = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            Flatten(),
            nn.Linear(64*6*6, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )
        self.train()
        
    def forward(self, x):
        x = self.model(x)
        return x

    def act(self, state, epsilon):
        with torch.no_grad():
            if random.random() > epsilon:
                q_value = self.forward(state)
                #print(q_value)
                action  = torch.argmax(q_value, dim=1)[0]
            else:
                action = random.randrange(self.num_actions)
        return action
