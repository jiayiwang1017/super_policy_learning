
#%%
import os
import sys
import csv
#%%
import numpy as np
import torch
from envs import ContinuousEnv
from agents import Policy
from utils import *
from collections import OrderedDict
from prox_fqe import fit_qpi_cv, fit_v0
from scipy.special import expit as scipy_expit
from rkhs_torch import _to_tensor


import argparse
parser = argparse.ArgumentParser()
# parser.add_argument('--linearity', type =str,  default= 'linear')
parser.add_argument('--n', type =int,  default= 200)
parser.add_argument('--T', type =int,  default= 200)
parser.add_argument('--tu', type =float,  default= 1.0)
parser.add_argument('--seed', type = int, default= 1)
args = parser.parse_args()




# In[142]:


episode_len    = args.T # e.g., 1,2,3,4,5
samp_size      = args.n # e.g., 256, 512, 1024,...
epsilon        = 0.4 # 0.4,0.2,0.1
device         = 'cpu'      # 'cuda:0', 'cuda:1',... 
test_samp_size = 5000 # e.g. 5000, 10000, 50000
# file_path      = 


# In[143]:


# %% Setup Continuous State Environment  t_u -> kappa_a -> mu_a
continuousParams = {'episode_len':episode_len, 'offline':True, 
                    'alpha_0':0, 'alpha_a':0.5, 'alpha_s':[0.5, 0.5], 
                    'mu_0':0,    'mu_a':-0.25,  'mu_s':[0.5, 0.5], 
                    'kappa_0':0, 'kappa_a':-0.5, 'kappa_s':[0.5,0.5], 
                    't_0':0,      't_u':args.tu,     't_s':[-0.5,-0.5]}
ContEnv= ContinuousEnv(continuousParams)


# In[144]:


train_cfg = {'episode_num':samp_size}
pfqe_option={'gamma_f':'auto', 'n_gamma_hs':20, 'n_alphas':30, 'cv':5}
train_batch = batch_data_collector(ContEnv, train_cfg, policy = None, seed=args.seed)


# In[145]:


### organize data 
Slist = []
Wlist = []
Zlist = []
Alist = []
Ylist = []

for t in range(episode_len):
    S = np.zeros([samp_size, 2])
    W = np.zeros([samp_size,1])
    Z = np.zeros([samp_size,1])
    A = np.zeros([samp_size,1])
    Y = np.zeros([samp_size,1])
    for i in range(samp_size):
        S[i,:] = train_batch[i][t]['state']['S']
        W[i,:] = train_batch[i][t]['state']['W']
        Z[i,:] = train_batch[i][t]['state']['Z']
        A[i,:] = train_batch[i][t]['action']
        Y[i,:] = train_batch[i][t]['reward']
    Slist.append(S)
    Wlist.append(W)
    Zlist.append(Z)
    Alist.append(A)
    Ylist.append(Y)


#### policy list
policy_super_lst = [None] * episode_len
policy_s_lst = [None] * episode_len
policy_sz_lst = [None] * episode_len
policy_dr_lst = [None] * episode_len
proxy_ds_lst = [None] * episode_len
proxy_dr_lst = [None] * episode_len
proxy_dsz_lst = [None] * episode_len
qpi_dr_lst = [None] * episode_len
qpi_ds_lst = [None] * episode_len
qpi_dsz_lst = [None] * episode_len
qpi_lst = [None] * episode_len
proj_alpha_lst = [None] * episode_len 
proj_alpha_sz_lst = [None] * episode_len
proj_alpha_s_lst = [None] * episode_len  

# In[146]:


import numpy as np
import torch
from rkhs_torch import ApproxRKHSIV, ApproxRKHSIVCV, _to_tensor
from utils import *

def fit_qpi_cv(S, W, Z, Rewards, Actions, action_space,option, device):
    W = torch.from_numpy(W)
    S = torch.from_numpy(S)
    Z = torch.from_numpy(Z)
    Actions = torch.from_numpy(Actions)
    Rewards = torch.from_numpy(Rewards)
    WS = torch.cat((W,S), dim=1)
    ZS = torch.cat((Z,S), dim=1)
    qpi = []
    #for action in action_space:
    for action in range( action_space.n):
        index = torch.flatten(Actions==action)
        qpi_action = ApproxRKHSIVCV(gamma_f=option['gamma_f'], 
                                    n_gamma_hs=option['n_gamma_hs'],
                                    n_alphas=option['n_alphas'],
                                    cv = option['cv'],
                                    # n_components= max(int(torch.sqrt(torch.sum(index)).detach().cpu()), 25), 
                                    n_components= int(2*np.sqrt(samp_size)), 
                                    device = device
        ).fit(
            WS[index,:], 
            Rewards[index],
            ZS[index,:]
        )
        qpi.append(qpi_action)
    return qpi


# In[147]:


def projection_cv(RootKf, X, y,  lam_seq):
    n_components = RootKf.shape[1]
    eye_n_comp = torch.eye(n_components, dtype=X.dtype, device=X.device)
    val_errors = np.zeros(lam_seq.shape)
    n = X.shape[0]
    eye_n= torch.eye(n, dtype=X.dtype, device=X.device)
    for k, lam in np.ndenumerate(lam_seq):
        lam_c = 1/(n*lam) 
        projS = lam_c * RootKf @ torch.linalg.pinv(eye_n_comp + lam_c * RootKf.T @ RootKf ) @ RootKf.T
        res =torch.div( (y - projS @ y) , (1 - torch.diag(projS)))
        val_errors[k] = torch.mean(torch.square(res))
    lam_select = lam_seq[np.argmin(val_errors)]
    alpha = torch.linalg.pinv( n * lam_select* eye_n_comp + RootKf.T @ RootKf ) @ RootKf.T @ y
    return lam_select, val_errors, alpha


# In[148]:


import gym
from gym import spaces
# from gym.utils import seeding
action_space = spaces.Discrete(2)
option={'n_component':20, 'gamma_f':'auto', 'n_gamma_hs':10, 'n_alphas':20, 'n_components':20, 'cv':5}

def policy_super(S, Z, A, qpi, proj_alpha_list):
    Storch = torch.tensor(S).reshape([-1,2])
    Ztorch = torch.tensor(Z).reshape([-1,1])
    Atorch = torch.tensor(A).reshape([-1,1])
    super_act = np.zeros(Ztorch.shape[0])
    for action in range(action_space.n):
        index = torch.flatten(Atorch==action)
        if torch.sum(index) ==0:
            continue
        else:
            projq = np.zeros([torch.sum(index), action_space.n])
            for take_action in range(action_space.n):
                ZStorch = torch.cat((Ztorch,Storch), dim=1)[index,:]
                # WStorch = torch.cat((Wtorch,Storch), dim=1)[index,:]
                X = qpi[action].transcondition.transform(ZStorch)
                projq[:,take_action] = qpi[action].featCond.transform(X) @ (proj_alpha_list[action][take_action]).reshape(-1)
            # projection_list.append(projq)
            super_act[index.numpy()] = np.argmax(projq,axis = 1)
    return super_act

def policy_sz(S, Z,  qpi, proj_alpha_sz_list):
    Storch = torch.tensor(S).reshape([-1,2])
    Ztorch = torch.tensor(Z).reshape([-1,1])
    projq = np.zeros([S.shape[0],  action_space.n])
    ZStorch = torch.cat((Ztorch,Storch), dim=1)
    for action in range(action_space.n):       
        X = qpi[action].transcondition.transform(ZStorch)
        projq[:,action] = qpi[action].featCond.transform(X) @ (proj_alpha_sz_list[action]).reshape(-1)
        # projection_list.append(projq)
    sz_act = np.argmax(projq,axis = 1)
    return sz_act



idx = episode_len - 1


# In[150]:


A = Alist[idx]
S = Slist[idx]
W = Wlist[idx]
Z = Zlist[idx]
Y = Ylist[idx]



qpi_lst[idx] = fit_qpi_cv(S = S, W = W, Z = Z, Rewards = Y, Actions = A, action_space = action_space,option = option, device = 'cpu')
qpi_dsz_lst[idx] = fit_qpi_cv(S = S, W = W, Z = Z, Rewards = Y, Actions = A, action_space = action_space,option = option, device = 'cpu')



lam_seq = np.exp(np.linspace(np.log(1e-05), np.log(1), num=30, axis=0))

# conditioned on expert's action
projection_list = []
proj_alpha_lst[idx] = []
for action in range( action_space.n):
    index = torch.flatten(torch.from_numpy(A)==action)
    projq = np.zeros([torch.sum(index),  action_space.n])
    proj_alpha = []
    for take_action in range( action_space.n):
        ZStorch = torch.cat((torch.from_numpy(Z),torch.from_numpy(S)), dim=1)[index,:]
        WStorch = torch.cat((torch.from_numpy(W),torch.from_numpy(S)), dim=1)[index,:]
        y = qpi_lst[idx][take_action].predict(WStorch)
        RootKf = qpi_lst[idx][action].RootKf
        lam_select, val_errors,  alpha = projection_cv(RootKf, ZStorch,  y,  lam_seq)
        print(lam_select)
        # print(val_errors)
        projq[:,take_action] = RootKf @ alpha.reshape([-1])
        proj_alpha.append(alpha)
    # projection_list.append(projq)
    (proj_alpha_lst[idx]).append(proj_alpha)


proj_alpha_sz_lst[idx] = []
for take_action in range( action_space.n):
    ZStorch = torch.cat((torch.from_numpy(Z),torch.from_numpy(S)), dim=1)[index,:]
    WStorch = torch.cat((torch.from_numpy(W),torch.from_numpy(S)), dim=1)[index,:]
    y = qpi_dsz_lst[idx][take_action].predict(WStorch)
    RootKf = qpi_lst[idx][action].RootKf
    lam_select, val_errors,  alpha = projection_cv(RootKf, ZStorch,  y,  lam_seq)
    print(lam_select)
    # print(val_errors)
    projq[:,take_action] = RootKf @ alpha.reshape([-1])
    (proj_alpha_sz_lst[idx]).append(alpha)



def new_res(R, S, W, qpi, take_action):
    take_action = take_action.reshape([-1,1])
    WStorch = torch.cat((torch.from_numpy(W),torch.from_numpy(S)), dim=1)
    Y = R
    for action in range(action_space.n):
        Y = Y + ((qpi[action].predict(WStorch)) * (take_action == action )).detach().numpy()
    return Y


# In[153]:


for idx in range(episode_len-2, -1, -1):
    print(idx)
    A = Alist[idx]
    S = Slist[idx]
    W = Wlist[idx]
    Z = Zlist[idx]
    R = Ylist[idx]


    take_action = policy_sz(Slist[idx+1],Zlist[idx+1], qpi_dsz_lst[idx+1], proj_alpha_sz_list= proj_alpha_sz_lst[idx+1])
    Y_dsz = new_res(R, Slist[idx+1], Wlist[idx+1], qpi_dsz_lst[idx+1], take_action = take_action)

    qpi_dsz_lst[idx] = fit_qpi_cv(S = S, W = W, Z = Z, Rewards = Y_dsz, Actions = A, action_space = action_space,option = option, device = 'cpu')


    take_action = policy_super(Slist[idx+1],Zlist[idx+1],Alist[idx+1], qpi_lst[idx+1], proj_alpha_list= proj_alpha_lst[idx+1])
    Y_super = new_res(R,  Slist[idx+1], Wlist[idx+1], qpi_lst[idx+1], take_action = take_action)
    qpi_lst[idx] = fit_qpi_cv(S = S, W = W, Z = Z, Rewards = Y_super, Actions = A, action_space = action_space,option = option, device = 'cpu')

    lam_seq = np.exp(np.linspace(np.log(1e-05), np.log(1), num=30, axis=0))
    proj_alpha_lst[idx] = []
    for action in range( action_space.n):
        index = torch.flatten(torch.from_numpy(A)==action)
        projq = np.zeros([torch.sum(index),  action_space.n])
        proj_alpha = []
        for take_action in range( action_space.n):
            ZStorch = torch.cat((torch.from_numpy(Z),torch.from_numpy(S)), dim=1)[index,:]
            WStorch = torch.cat((torch.from_numpy(W),torch.from_numpy(S)), dim=1)[index,:]
            y = qpi_lst[idx][take_action].predict(WStorch)
            RootKf = qpi_lst[idx][action].RootKf
            lam_select, val_errors,  alpha = projection_cv(RootKf, ZStorch,  y,  lam_seq)
            print(lam_select)
            # print(val_errors)
            projq[:,take_action] = RootKf @ alpha.reshape([-1])
            proj_alpha.append(alpha)
        # projection_list.append(projq)
        (proj_alpha_lst[idx]).append(proj_alpha)

    proj_alpha_sz_lst[idx] = []
    for take_action in range(action_space.n):
        ZStorch = torch.cat((torch.from_numpy(Z),torch.from_numpy(S)), dim=1)[index,:]
        WStorch = torch.cat((torch.from_numpy(W),torch.from_numpy(S)), dim=1)[index,:]
        y = qpi_dsz_lst[idx][take_action].predict(WStorch)
        RootKf = qpi_lst[idx][action].RootKf
        lam_select, val_errors,  alpha = projection_cv(RootKf, ZStorch,  y,  lam_seq)
        print(lam_select)
        # print(val_errors)
        projq[:,take_action] = RootKf @ alpha.reshape([-1])
        (proj_alpha_sz_lst[idx]).append(alpha)
   


# %% Setup Continuous State Environment  t_u -> kappa_a -> mu_a
continuousParams = {'episode_len':episode_len, 'offline':False, 
                    'alpha_0':0, 'alpha_a':0.5, 'alpha_s':[0.5, 0.5], 
                    'mu_0':0,    'mu_a':-0.25,  'mu_s':[0.5, 0.5], 
                    'kappa_0':0, 'kappa_a':-0.5, 'kappa_s':[0.5,0.5], 
                    't_0':0,      't_u':args.tu,     't_s':[-0.5,-0.5]}
ContEnv= ContinuousEnv(continuousParams)





from scipy.special import expit
##### Evaluation ######
def MC_evaluator(env,  policy_name,  config, seed=0, policy_list= None):
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
                exp_act = np.array(env.exp_act)
                if policy_name == 'super':
                    action = policy_super(obs['S'], obs['Z'], exp_act, qpi_lst[t], proj_alpha_lst[t])[0].astype(int)
                if policy_name == 'sz':
                    action = policy_sz(obs['S'], obs['Z'], qpi_dsz_lst[t], proj_alpha_sz_lst[t])[0].astype(int)
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


# In[156]:


MC_cfg={'episode_num':test_samp_size, 'verbose':False}


val_dsz,_ , _, _ = MC_evaluator(ContEnv, policy_name = 'sz',  config = MC_cfg, seed=0)



val_super,_ , _, _ = MC_evaluator(ContEnv, policy_name = 'super',  config = MC_cfg, seed=0)


values = np.array([val_dsz, val_super])

print({"common":val_dsz, "super":val_super})


