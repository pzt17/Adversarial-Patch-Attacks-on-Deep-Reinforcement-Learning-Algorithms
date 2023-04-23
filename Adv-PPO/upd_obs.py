import numpy as np
import copy

def upd_obs(test_obs, patch, pos_x, pos_y):


    patch_shape = patch.shape[0]
    batch_size = test_obs.shape[0]
    for i in range(batch_size):
    	for rgb in range(3):
    		test_obs[i, pos_x: pos_x + patch_shape, pos_y: pos_y + patch_shape, rgb] = patch[:, :, rgb]
    	    	    
    return test_obs
    
def mask_obs(test_obs, patch, pos_x, pos_y):


    patch_shape = patch.shape[0]
    batch_size = test_obs.shape[0]
    for i in range(batch_size):
    	for rgb in range(3):
    		test_obs[i, pos_x: pos_x + patch_shape, pos_y: pos_y + patch_shape, rgb] = 255.0
    	    	    
    return test_obs
    
def syn_obs(gen_obs2, mask_obs, patch_shape, pos_x, pos_y, pos_x2, pos_y2):

    batch_size = gen_obs2.shape[0]
    gen_obs = gen_obs2.clone()
    
    half_size = int(batch_size/2)
    for i in range(half_size):
    	for rgb in range(3):
    		gen_obs[i, pos_x[i]: pos_x[i] + patch_shape, pos_y[i]: pos_y[i] + patch_shape, rgb] = mask_obs[i, pos_x[i]: pos_x[i] + patch_shape, pos_y[i]: pos_y[i] + patch_shape, rgb]
    		gen_obs[i+ half_size, pos_x2[i]: pos_x2[i] + patch_shape, pos_y2[i]: pos_y2[i] + patch_shape, rgb] = mask_obs[i + half_size, pos_x2[i]: pos_x2[i] + patch_shape, pos_y2[i]: pos_y2[i] + patch_shape, rgb]
    		
    return gen_obs
    
def syn_obs_eval(gen_obs2, mask_obs, patch_shape, pos_x, pos_y):

    batch_size = gen_obs2.shape[0]
    
    half_size = int(batch_size/2)
    
    for i in range(batch_size):
    	for rgb in range(3):
    		gen_obs2[i, pos_x: pos_x+ patch_shape, pos_y: pos_y + patch_shape, rgb] = mask_obs[i, pos_x: pos_x+ patch_shape, pos_y: pos_y + patch_shape, rgb]
    		
    		
    return gen_obs2

def ran_mask_obs(test_obs, patch):


    patch_shape = patch.shape[0]
    batch_size = test_obs.shape[0]
    pos_x = np.random.randint(0, 64 - patch_shape + 1)
    pos_y = np.random.randint(0, 64 - patch_shape + 1)
    for i in range(batch_size):
    	for rgb in range(3):
    		test_obs[i, pos_x: pos_x + patch_shape, pos_y: pos_y + patch_shape, rgb] = 0.0
    	    	    
    return test_obs
    

def patch_obs(states, patch, pos_x, pos_y):
	patch_size = patch.shape[1]
	batch_size = states.shape[0]
	states = copy.deepcopy(states)
	for i in range(batch_size):
		for rgb in range(3):
			states[i, pos_x:pos_x + patch_size, pos_y:pos_y + patch_size, rgb] = patch[:, :, rgb]
	return states
	
def update_patch(states_grad, patch, pos_x, pos_y):
	grad_sum = np.zeros((states_grad.shape[1], states_grad.shape[2], states_grad.shape[3]))
	patch_size = patch.shape[1]
	batch_size = states_grad.shape[0]
	for i in range(batch_size):
		for rgb in range(3):
			grad_sum[pos_x: pos_x + patch_size, pos_y: pos_y + patch_size , rgb] += states_grad[i, pos_x: pos_x + patch_size, pos_y: pos_y + patch_size , rgb]
	
	patch -= 1.0 * np.sign(grad_sum[pos_x: pos_x + patch_size, pos_y: pos_y + patch_size, :])
	patch = np.clip(patch, 0, 255)
	return patch

