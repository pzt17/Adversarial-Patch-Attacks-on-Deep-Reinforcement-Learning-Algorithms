from collections import deque
import argparse
import os
import time, datetime
import torch
import numpy as np
import random

from procgen import ProcgenEnv
from vec_env import VecExtractDictObs
from vec_env import VecMonitor
from vec_env import VecNormalize
from util import logger

from policies import ImpalaCNN
from ppo import PPO
from square import init_patch_square

def rollout_one_step(agent, env, obs, steps, env_max_steps=1000, pgd_epsilon=None, gwc_epsilon=None):
    assert not (pgd_epsilon and gwc_epsilon)
    # Step once.
    if pgd_epsilon:
        action = agent.batch_act_pgd(obs, pgd_epsilon)
    elif gwc_epsilon:
        action = agent.batch_act_gwc(obs, gwc_epsilon)
    else:
        action = agent.batch_act(obs)
    #print(action)
    new_obs, reward, done, infos = env.step(action)
    env.render()
    steps += 1
    reset = steps == env_max_steps
    steps[done] = 0

    # Save experience.
    agent.batch_observe(
        batch_obs=new_obs,
        batch_reward=reward,
        batch_done=done,
        batch_reset=reset,
    )

    # Get rollout statistics.
    epinfo = []
    for info in infos:
        maybe_epinfo = info.get('episode')
        if maybe_epinfo:
            epinfo.append(maybe_epinfo)

    return new_obs, steps, epinfo

def rollout_one_step_oppor(agent, env, obs, steps, env_max_steps=1000, pgd_epsilon=None, gwc_epsilon=None, random_ep = 0.3, num_envs = 1):
    assert not (pgd_epsilon and gwc_epsilon)
    # Step once.
    if pgd_epsilon:
        action = agent.batch_act_pgd(obs, pgd_epsilon)
    elif gwc_epsilon:
        action = agent.batch_act_gwc(obs, gwc_epsilon)
    else:
        action = agent.batch_act(obs)
        #print('ac1', action)
    #print(action)
    
    if random.random() < random_ep:
    	action = np.zeros((num_envs))
    	for i in range(num_envs):
    		action[i] = np.int(env.action_space.sample())
    
    new_obs, reward, done, infos = env.step(action)
    env.render()
    steps += 1
    reset = steps == env_max_steps
    steps[done] = 0

    # Save experience.
    agent.batch_observe(
        batch_obs=new_obs,
        batch_reward=reward,
        batch_done=done,
        batch_reset=reset,
    )

    # Get rollout statistics.
    epinfo = []
    for info in infos:
        maybe_epinfo = info.get('episode')
        if maybe_epinfo:
            epinfo.append(maybe_epinfo)

    return new_obs, steps, epinfo
