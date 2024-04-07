# basic env
import os
import copy
import random
import datetime
import traceback

# plot
# import matplotlib
# import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.nn.functional as F
import torch.cuda

# for replay buffer
from collections import namedtuple

# common function
def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# batch data collector
def batch_data_collector(env, config, policy = None, seed=0):
    # config = {'episode_len': int, 'episode_num': int}, etc.
    if policy is None:
        env.params['offline'] = True
    else:
        env.params['offline'] = False

    set_seeds(seed)
    Episodes = []
    for n in range(config['episode_num']):
        Episodes.append([])
        try:
            # need seed maybe
            obs = env.reset() # may need seed
            for t in range(env.params['episode_len']):
                action = env.act_last if policy is None else policy.eps_greedy(obs)
                Episodes[n].append({'state':obs, 'action':action, 'reward': 0.})
                obs, r, done, _ = env.step(action)
                Episodes[n][t]['reward'] = r
                if done:
                    break
        except KeyboardInterrupt:
            break
        except:
            traceback.print_exc()
            break
    # close env
    env.close()

    print(f"{env.name} | Episode_num: {config['episode_num']} | Episode_len: {env.params['episode_len']}", flush=True)
    return Episodes

def batch_cat(Episodes, action_space, observation_space, device, verbose=False):
    # Init concatenated Episodes data to torch.Tensor
    Episodes_cat = {}
    Episodes_cat['action'] = torch.empty((len(Episodes),1,len(Episodes[0])), dtype=torch.int8, device=device)
    Episodes_cat['reward'] = torch.empty((len(Episodes),1,len(Episodes[0])), dtype=torch.float64, device=device)
    for key, value in observation_space.items():
        Episodes_cat[key] = torch.empty((len(Episodes),) + value._shape + (len(Episodes[0]),), dtype=torch.float64, device=device)

    for n in range(len(Episodes)):
        for t in range(len(Episodes[0])):
            Episodes_cat['action'][n,:,t] = Episodes[n][t]['action']
            Episodes_cat['reward'][n,:,t] = Episodes[n][t]['reward']
            for key, value in observation_space.items():
                if type(Episodes[n][t]['state'][key]) is np.ndarray:
                    Episodes_cat[key][n,:,t]  = torch.from_numpy(Episodes[n][t]['state'][key]).to(Episodes_cat[key])
                else:
                    Episodes_cat[key][n,:,t]  = Episodes[n][t]['state'][key]
    
    if verbose:
        for t in range(len(Episodes[0])):
            print(f"\t\tStep{t} ", end='', flush=True) 
            for action in range(action_space.start, action_space.start + action_space.n):
                count_action = torch.flatten(Episodes_cat['action'][:,:,t]==action).sum().detach().cpu()
                print(f"| action {action}: {count_action} ", end='', flush=True)
            print('', flush=True)

    return Episodes_cat

# Monte-Carlo Evaluation of a given fixed policy (not an agent)
def MC_evaluator(env, policy, config, seed=0):
    # config = {'episode_num': int, 'verbose': bool}, etc.
    set_seeds(seed)
    
    Rewards = []
    Initial_S = []
    Initial_W = []
    for n in range(config['episode_num']):
        try:
            # need seed maybe
            reward = 0
            obs = env.reset() # may need seed
            Initial_S.append(obs['S'])
            Initial_W.append(obs['W'])
            for t in range(env.params['episode_len']):
                action = policy.eps_greedy(obs)
                obs, r, done, _ = env.step(action)
                reward += r
                if done:
                    break
            Rewards.append(reward)
            if config['verbose']:
                print(f"Monte Carlo Episode: {t} | Reward: {reward:.1f}")
        except KeyboardInterrupt:
            break
        except:
            traceback.print_exc()
            break
    # close env
    env.close()

    mean_reward = np.mean(Rewards)
    print(f"Mean reward: {mean_reward}")

    return mean_reward, Rewards, Initial_S, Initial_W
    

# plot

# %%
