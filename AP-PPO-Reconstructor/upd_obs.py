import numpy as np
import copy

def upd_obs(test_obs, patch, pos_x, pos_y):


    patch_shape = patch.shape[0]
    for rgb in range(3):
    	test_obs[0, pos_x: pos_x + patch_shape, pos_y: pos_y + patch_shape, rgb] = patch[:, :, rgb]
    	    	    
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

