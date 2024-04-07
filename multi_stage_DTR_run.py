import os
import sys
import csv
#%%
import numpy as np
import torch
from envs import ContinuousEnv
# from agents import Policy
from utils import *
from collections import OrderedDict
from prox_fqe import fit_qpi_cv, fit_v0
from scipy.special import expit as scipy_expit
from torch.special import expit as torch_expit
from rkhs_torch import _to_tensor


import argparse
parser = argparse.ArgumentParser()
# parser.add_argument('--linearity', type =str,  default= 'linear')
parser.add_argument('--n', type =int,  default= 200)
parser.add_argument('--T', type =int,  default= 2)
parser.add_argument('--tu', type =float,  default= 1.0)
parser.add_argument('--seed', type = int, default= 1)
args = parser.parse_args()



episode_len    = args.T # e.g., 1,2,3,4,5
samp_size      = args.n # e.g., 256, 512, 1024,...
epsilon        = 0.4 # 0.4,0.2,0.1
device         = 'cpu'      # 'cuda:0', 'cuda:1',... 
test_samp_size = 10000 # e.g. 5000, 10000, 50000
tu = args.tu
# file_path      = 



# %% Setup Continuous State Environment  t_u -> kappa_a -> mu_a
continuousParams = {'episode_len':episode_len, 'offline':True, 
                    'alpha_0':0, 'alpha_a':0.5, 'alpha_s':[0.5, 0.5], 
                    'mu_0':0,    'mu_a':-0.25,  'mu_s':[0.5, 0.5], 
                    'kappa_0':0, 'kappa_a':-0.5, 'kappa_s':[0.5,0.5], 
                    't_0':0,      't_u':tu,     't_s':[-0.5,-0.5]}
ContEnv= ContinuousEnv(continuousParams)


# In[144]:


train_cfg = {'episode_num':samp_size}
pfqe_option={'gamma_f':'auto', 'n_gamma_hs':20, 'n_alphas':30, 'cv':5}
train_batch = batch_data_collector(ContEnv, train_cfg, policy = None, seed=args.seed)

### organize data 
Slist = []
Olist = []
Aoldlist = []
Ahislist = []
Wlist = []
Zlist = []
Alist = []
Ylist = []
Ulist = []

for t in range(episode_len):
    S = np.zeros([samp_size, 2])
    W = np.zeros([samp_size,1])
    U = np.zeros([samp_size,1])
    Z = np.zeros([samp_size,1])
    A = np.zeros([samp_size,1])
    Y = np.zeros([samp_size,1])
    for i in range(samp_size):
        S[i,:] = train_batch[i][t]['state']['S']
        W[i,:] = train_batch[i][t]['state']['W']
        Z[i,:] = train_batch[i][t]['state']['Z']
        A[i,:] = train_batch[i][t]['action']
        Y[i,:] = train_batch[i][t]['reward']
        U[i,:] = train_batch[i][t]['state']['U']
    if t == 0:
        O = S
        Ahis = A
    else:
        O = np.concatenate([O, S], axis = 1)
        Ahis = np.concatenate([Ahis, A], axis = 1)
    Slist.append(S)
    Wlist.append(W)
    Zlist.append(Z)
    Alist.append(A)
    Ylist.append(Y)
    Ulist.append(U)
    Olist.append(O)
    if t==0:
        Aoldlist.append(np.array([]).reshape(samp_size,0))
    if t>0:
        Aoldlist.append(np.delete(Ahis, 1, Ahis.shape[1]-1))
    Ahislist.append(Ahis)


#### policy list
policy_super_lst = [None] * episode_len
policy_s_lst = [None] * episode_len
policy_sz_lst = [None] * episode_len
policy_dr_lst = [None] * episode_len
proxy_ds_lst = [None] * episode_len
proxy_dr_lst = [None] * episode_len
proxy_dsz_lst = [None] * episode_len
qpi_dr_lst = [None] * episode_len
qpi_lst = [None] * episode_len
qpi_ds_lst = [None] * episode_len
qpi_dsz_lst = [None] * episode_len
qpi_super_lst = [None] * episode_len
proj_alpha_super_lst = [None] * episode_len 
proj_alpha_s_lst = [None] * episode_len 
proj_alpha_sz_lst = [None] * episode_len 


import itertools

def bin_array(N, dtype):
    return (np.arange(1<<N, dtype=dtype)[:, None] >> np.arange(N, dtype=dtype)[::-1]) & 0b1



import numpy as np
import torch
from rkhs_torch import ApproxRKHSIV, ApproxRKHSIVCV, _to_tensor
from utils import *




def fit_qpi_cv(S, W, Z, Rewards, Ahisactions, option, device):
    W = torch.from_numpy(W)
    S = torch.from_numpy(S)
    Z = torch.from_numpy(Z)
    Ahisactions = torch.from_numpy(Ahisactions)
    Rewards = torch.from_numpy(Rewards)
    WS = torch.cat((W,S), dim=1)
    ZS = torch.cat((Z,S), dim=1)
    qpi = []
    #for action in action_space:
    # for action in range(action_space.start, action_space.start + action_space.n):
    action_combinations = bin_array(Ahisactions.shape[1], int)
    for actions in action_combinations:
        # index = torch.flatten(Ahisactions==actions)
        index = (torch.sum(torch.square(Ahisactions - actions), axis = 1)==0)
        qpi_action = ApproxRKHSIVCV(gamma_f=option['gamma_f'], 
                                    n_gamma_hs=option['n_gamma_hs'],
                                    n_alphas=option['n_alphas'],
                                    cv = option['cv'],
                                    # n_components= max(int(torch.sqrt(torch.sum(index)).detach().cpu()), 25), 
                                    # n_components= int(2*np.sqrt(samp_size)), 
                                    n_components= min(int(2*np.sqrt(samp_size)), sum(index)), 
                                    device = device
        ).fit(
            WS[index,:], 
            Rewards[index],
            ZS[index,:]
        )
        qpi.append(qpi_action)
    return qpi



import gym
from gym import spaces
from gym.utils import seeding
action_space = spaces.Discrete(2)
option={'n_component':20, 'gamma_f':'auto', 'n_gamma_hs':10, 'n_alphas':20, 'n_components':20, 'cv':5}


def policy_super_last(S, Z, Ahisactions, Aoldactions, proj_alpha_super_last_list):
    # A_encode = encoder.transform(A)
    # A_encode = A_encode.toarray()
    action_combinations = bin_array(Ahisactions.shape[1], int)
    super_act = np.zeros(Z.shape[0])
    projq = np.zeros([S.shape[0] , action_space.n])
    def model_index_fn(x):
        index = np.where((np.sum(np.square(action_combinations - x), axis = 1)==0))
        return index[0][0]
    for take_action in range(action_space.n):
        # ZSA = np.concatenate([Z, S, Ahisactions], axis = 1)
        ZSA = np.concatenate([S, Ahisactions], axis = 1)
        take_actions = np.concatenate((Aoldactions, take_action * np.ones([Aoldactions.shape[0],1])), axis = 1)
        model_index = np.apply_along_axis(model_index_fn,  1, take_actions)
        # print(model_index)
        for i in range(S.shape[0]):
            projq[i,take_action] = proj_alpha_super_last_list[model_index[i]].predict(ZSA[i,:].reshape([1,-1])).reshape(-1)
    super_act = np.argmax(projq,axis = 1)
    return super_act



def policy_super(S, Z, Ahisactions, Aoldactions, proj_alpha_super_list):
    super_act = np.zeros(Z.shape[0])
    projq = np.zeros([S.shape[0] , action_space.n])
    def model_index_fn(x):
        index = np.where((np.sum(np.square(action_combinations - x), axis = 1)==0))
        return index[0][0]
    for take_action in range(action_space.n):
        if Ahisactions.shape[1]==1:
            exp_old_and_take_action = np.ones((Ahisactions.shape[0], 1))* take_action
            take_old_and_exp_action = Ahisactions
        else:
            exp_old_and_take_action = np.concatenate([Ahisactions[:,:-1], np.ones((Ahisactions.shape[0], 1))* take_action], axis = 1)
            take_old_and_exp_action = np.concatenate([Aoldactions, Ahisactions[:,-1]], axis = 1)
        model_index = np.apply_along_axis(model_index_fn,  1, exp_old_and_take_action)
        # print(model_index)
        # ZSA = np.concatenate([Z, S, take_old_and_exp_action], axis = 1)
        ZSA = np.concatenate([S, take_old_and_exp_action], axis = 1)
        for i in range(S.shape[0]):
            projq[i,take_action] = proj_alpha_super_list[model_index[i]].predict(ZSA[i,:].reshape([1,-1])).reshape(-1)
    super_act = np.argmax(projq,axis = 1)
    return super_act



def policy_sz(S, Z,  Aoldactions, proj_alpha_list):
    # A_encode = encoder.transform(A)
    # A_encode = A_encode.toarray()
    # action_combinations = bin_array(Ahisactions.shape[1]+1, int)
    super_act = np.zeros(S.shape[0])
    projq = np.zeros([S.shape[0] , action_space.n])
    # def model_index_fn(x):
    #     index = np.where((np.sum(np.square(action_combinations - x), axis = 1)==0))
    #     return index[0][0]
    for take_action in range(action_space.n):
        ZSAold = np.concatenate([Z, S, Aoldactions], axis = 1)
        projq[:,take_action] = proj_alpha_list[take_action].predict(ZSAold).reshape(-1)
        # take_actions = np.concatenate((Aoldactions, take_action * np.ones([Aoldactions.shape[0],1])), axis = 1)
        # model_index = np.apply_along_axis(model_index_fn,  1, take_actions)
        # print(model_index)
        # for i in range(S.shape[0]):
        #     projq[i,take_action] = proj_alpha_list[model_index[i]].predict(ZSAold[i,:].reshape([1,-1])).reshape(-1)
    super_act = np.argmax(projq,axis = 1)
    return super_act


def policy_s(S, Z,  Aoldactions, proj_alpha_list):
    # A_encode = encoder.transform(A)
    # A_encode = A_encode.toarray()
    # action_combinations = bin_array(Ahisactions.shape[1]+1, int)
    super_act = np.zeros(S.shape[0])
    projq = np.zeros([S.shape[0] ,  action_space.n])
    # def model_index_fn(x):
    #     index = np.where((np.sum(np.square(action_combinations - x), axis = 1)==0))
    #     return index[0][0]
    for take_action in range(action_space.n):
        SAold = np.concatenate([S, Aoldactions], axis = 1)
        projq[:,take_action] = proj_alpha_list[take_action].predict(SAold).reshape(-1)
        # take_actions = np.concatenate((Aoldactions, take_action * np.ones([Aoldactions.shape[0],1])), axis = 1)
        # model_index = np.apply_along_axis(model_index_fn,  1, take_actions)
        # print(model_index)
        # for i in range(S.shape[0]):
        #     projq[i,take_action] = proj_alpha_list[model_index[i]].predict(SAold[i,:].reshape([1,-1])).reshape(-1)
    super_act = np.argmax(projq,axis = 1)
    return super_act



def policy_opt(S, U, qpi):
    Storch = torch.tensor(S).reshape([-1,2])
    Utorch = torch.tensor(U).reshape([-1,1])
    Qm = np.zeros([S.shape[0], action_space.n])
    UStorch = torch.cat((Utorch,Storch), dim=1)
    for action in range( action_space.n):
        Qm[:,action] = qpi[action].predict(UStorch).reshape(-1)
    take_action = np.argmax(Qm,axis = 1)
    return take_action


from sklearn.linear_model import LinearRegression
import gym
from gym import spaces

idx = episode_len - 1
S = Olist[idx]
Z = Zlist[0]
W = Wlist[idx]
Y = Ylist[idx]
Ahisactions = Ahislist[idx]
Aoldactions = Aoldlist[idx]
# print(Aoldactions)

action_space = spaces.Discrete(2)
action_combinations = bin_array(Ahisactions.shape[1], int)
# ZSA = np.concatenate((Z, S, Ahisactions), axis = 1)
ZSA = np.concatenate((S, Ahisactions), axis = 1)
ZS = np.concatenate((Z, S), axis = 1)
WS = np.concatenate((W, S), axis = 1)
option={'n_component':20, 'gamma_f':'auto', 'n_gamma_hs':10, 'n_alphas':20, 'n_components':20, 'cv':5}
qpi_super_last = fit_qpi_cv(S = S, W = W, Z = Z, Rewards = Y, Ahisactions = Ahisactions, option = option, device = 'cpu')

if idx==episode_len-1:
    qpi_super_lst[idx] = [None] * (action_combinations.shape[0])
    for k, take_actions in enumerate(action_combinations):
        qpi_super_lst[idx][k] = qpi_super_last

qpi_ds_lst[idx] = fit_qpi_cv(S = S, W = W, Z = Z, Rewards = Y, Ahisactions = Ahisactions, option = option, device = 'cpu')

qpi_dsz_lst[idx] = fit_qpi_cv(S = S, W = W, Z = Z, Rewards = Y, Ahisactions = Ahisactions, option = option, device = 'cpu')

qpi_lst[idx] = fit_qpi_cv(S = Slist[idx], W = Ulist[idx], Z = Ulist[idx], Rewards = Y, Ahisactions = Alist[idx], option = option, device = 'cpu')

proj_alpha_super_last_list = []
for k, take_actions in enumerate(action_combinations):
    # index = (torch.sum(torch.square(Ahisactions - take_actions), axis = 1)==0)
    # WS = np.concatenate((W, S), axis = 1)
    WStorch = torch.cat((torch.from_numpy(W),torch.from_numpy(S)), dim=1)
    y = qpi_super_last[k].predict(WStorch)
    y = y.detach().numpy()
    # print(np.min(y))
    reg = LinearRegression().fit(ZSA, y)
    proj_alpha_super_last_list.append(reg)


proj_alpha_sz_lst[idx] = []
ZSAold = np.concatenate((Z, S, Aoldactions), axis = 1)
for take_action in range(action_space.start, action_space.start + action_space.n):
    # WS = np.concatenate((W, S), axis = 1)
    WStorch = torch.cat((torch.from_numpy(W),torch.from_numpy(S)), dim=1)
    def model_index_fn(x):
        index = np.where((np.sum(np.square(action_combinations - x), axis = 1)==0))
        return index[0][0]
    temp_actions = np.concatenate([Aoldactions, np.ones((Z.shape[0], 1))* take_action], axis = 1)
    model_index1 = np.apply_along_axis(model_index_fn,  1, temp_actions)
    y = np.zeros([samp_size,1])
    for i in range(samp_size):
        temp = qpi_dsz_lst[idx][model_index1[i]].predict(WStorch[i,:].reshape([1,-1]))
        y[i,:] = temp.detach().numpy()
    # print(np.min(y))
    reg = LinearRegression().fit(ZSAold, y)
    proj_alpha_sz_lst[idx].append(reg)


proj_alpha_s_lst[idx] = []
SAold = np.concatenate((S, Aoldactions), axis = 1)
for take_action in range(action_space.start, action_space.start + action_space.n):
    # WS = np.concatenate((W, S), axis = 1)
    WStorch = torch.cat((torch.from_numpy(W),torch.from_numpy(S)), dim=1)
    def model_index_fn(x):
        index = np.where((np.sum(np.square(action_combinations - x), axis = 1)==0))
        return index[0][0]
    temp_actions = np.concatenate([Aoldactions, np.ones((Z.shape[0], 1))* take_action], axis = 1)
    model_index1 = np.apply_along_axis(model_index_fn,  1, temp_actions)
    y = np.zeros([samp_size,1])
    for i in range(samp_size):
        temp = qpi_ds_lst[idx][model_index1[i]].predict(WStorch[i,:].reshape([1,-1]))
        y[i,:] = temp.detach().numpy()
    # print(np.min(y))
    reg = LinearRegression().fit(SAold, y)
    proj_alpha_s_lst[idx].append(reg)





def next_optim_res(take_actions, idx, policy, proj_alpha_list):
    Ahisactions_next_step = np.concatenate( [np.ones((Ahisactions.shape[0], 1))* take_actions, Alist[idx+1]], axis = 1)
    # Aoldactions_next_step = Ahisactions
    Aoldactions_next_step = Aoldlist[idx+1]
    action_combinations_next_step = bin_array(Ahislist[idx+1].shape[1], int)
    # print(action_combinations_next_step)
    def model_index_fn_next(x):
        index = np.where((np.sum(np.square(action_combinations_next_step - x), axis = 1)==0))
        return index[0][0]

    q_model_index_next_step = np.apply_along_axis(model_index_fn_next,  1, Ahisactions_next_step)
    # print(q_model_index_next_step)

    take_action_next_step = policy(Olist[idx+1], Z, Ahisactions_next_step, Aoldactions_next_step, proj_alpha_list)
    # print(take_action_next_step)
    q_model_inside_index_next_step = np.apply_along_axis(model_index_fn_next,  1, np.concatenate([Ahisactions, take_action_next_step.reshape([-1,1])], axis = 1))
    # print(q_model_inside_index_next_step)
    qnext = np.zeros([samp_size,1])
    WStorch_next = torch.cat((torch.from_numpy(Wlist[idx+1]),torch.from_numpy(Olist[idx+1])), dim=1)
    for i in range(samp_size):
        next_q_optim = (qpi_super_lst[idx+1][q_model_index_next_step[i]][q_model_inside_index_next_step[i]].predict(WStorch_next[i,:].reshape([1,-1])))[0]
        qnext[i,:] =  next_q_optim.detach().numpy()
    return qnext



def next_optim_res_others(idx, policy, proj_alpha_list, qpi_lst):
    # Ahisactions_next_step = np.concatenate( [np.ones((Ahisactions.shape[0], 1))* take_actions, Alist[idx+1]], axis = 1)
    Aoldactions_next_step = Aoldlist[idx+1]
    action_combinations_next_step = bin_array(Ahislist[idx+1].shape[1], int)
    # print(action_combinations_next_step)
    def model_index_fn_next(x):
        index = np.where((np.sum(np.square(action_combinations_next_step - x), axis = 1)==0))
        return index[0][0]

    # q_model_index_next_step = np.apply_along_axis(model_index_fn_next,  1, Ahisactions_next_step)
    # print(q_model_index_next_step)

    take_action_next_step = policy(Olist[idx+1], Z, Aoldactions_next_step, proj_alpha_list)
    # print(take_action_next_step)
    q_model_inside_index_next_step = np.apply_along_axis(model_index_fn_next,  1, np.concatenate([Aoldactions_next_step, take_action_next_step.reshape([-1,1])], axis = 1))
    # print(q_model_inside_index_next_step)
    qnext = np.zeros([samp_size,1])
    WStorch_next = torch.cat((torch.from_numpy(Wlist[idx+1]),torch.from_numpy(Olist[idx+1])), dim=1)
    for i in range(samp_size):
        next_q_optim = (qpi_lst[idx+1][q_model_inside_index_next_step[i]].predict(WStorch_next[i,:].reshape([1,-1])))[0]
        qnext[i,:] =  next_q_optim.detach().numpy()
    return qnext





for idx in range(episode_len-2, -1, -1):
# idx = episode_len - 1
    S = Olist[idx]
    Z = Zlist[0]
    W = Wlist[idx]
    R = Ylist[idx]
    Ahisactions = Ahislist[idx]
    Aoldactions = Aoldlist[idx]

    action_space = spaces.Discrete(2)
    action_combinations = bin_array(Ahisactions.shape[1], int)
    # ZSA = np.concatenate((Z, S, Ahisactions), axis = 1)
    ZSA = np.concatenate((S, Ahisactions), axis = 1)
    ZS = np.concatenate((Z, S), axis = 1)
    WS = np.concatenate((W, S), axis = 1)
    option={'n_component':20, 'gamma_f':'auto', 'n_gamma_hs':10, 'n_alphas':20, 'n_components':20, 'cv':5}

    qpi_super_lst[idx] = [None] * (action_combinations.shape[0])

    for k, take_actions in enumerate(action_combinations):
        if idx == episode_len-2:
            qnext = next_optim_res(take_actions, idx, policy = policy_super_last, proj_alpha_list= proj_alpha_super_last_list)
        else:
            qnext = next_optim_res(take_actions, idx, policy = policy_super, proj_alpha_list= proj_alpha_super_lst[idx+1])
        Y = R + qnext
        qpi_super_lst[idx][k] = fit_qpi_cv(S = S, W = W, Z = Z, Rewards = Y, Ahisactions = Ahisactions, option = option, device = 'cpu')


    qnext_sz = next_optim_res_others(idx, policy = policy_sz, proj_alpha_list= proj_alpha_sz_lst[idx+1], qpi_lst= qpi_dsz_lst)
    Y = R + qnext_sz
    qpi_dsz_lst[idx] = fit_qpi_cv(S = S, W = W, Z = Z, Rewards = Y, Ahisactions = Ahisactions, option = option, device = 'cpu')


    qnext_s = next_optim_res_others(idx, policy = policy_s, proj_alpha_list= proj_alpha_s_lst[idx+1], qpi_lst= qpi_ds_lst)
    Y = R + qnext_s
    qpi_ds_lst[idx] = fit_qpi_cv(S = S, W = W, Z = Z, Rewards = Y, Ahisactions = Ahisactions, option = option, device = 'cpu')



    
    take_action = policy_opt(Slist[idx+1],Ulist[idx+1], qpi_lst[idx+1])
    take_action = take_action.reshape([-1,1])


    def new_res(R, S, W, qpi, take_action):
        take_action = take_action.reshape([-1,1])  
        Y = R
        WStorch = torch.cat((torch.from_numpy(W),torch.from_numpy(S)), dim=1)
        for action in range(action_space.start, action_space.start + action_space.n):
            Y = Y + ((qpi[action].predict(WStorch)) * (take_action == action )).detach().numpy()
        return Y

    Y_super = new_res(R, Slist[idx+1], Wlist[idx+1], qpi_lst[idx+1], take_action = take_action)

    qpi_lst[idx] =  fit_qpi_cv(S = Slist[idx], W = Ulist[idx], Z = Ulist[idx], Rewards = Y_super, Ahisactions = Alist[idx], option = option, device = 'cpu')
            
    proj_alpha_super_lst[idx] = []
    # old_action_combination = bin_array(Ahisactions.shape[1]-1, int)
    for k, exp_old_and_take_action in enumerate(action_combinations):
        # WS = np.concatenate((W, S), axis = 1)
        if exp_old_and_take_action.shape[0] == 1:
            exp_his_actions = Ahisactions[:,-1].reshape([-1,1])
            take_actions = np.ones((Ahisactions.shape[0], 1))* exp_old_and_take_action[-1]
        else:    
            exp_his_actions = np.concatenate([np.ones((Ahisactions.shape[0], 1))* exp_old_and_take_action[:-1],Ahisactions[:,-1].reshape([-1,1])], axis = 1)
            take_actions = np.concatenate([Aoldactions.reshape([-1,1]),np.ones((Ahisactions.shape[0], 1))* exp_old_and_take_action[-1]], axis = 1)
        WStorch = torch.cat((torch.from_numpy(W),torch.from_numpy(S)), dim=1)
        def model_index_fn(x):
            index = np.where((np.sum(np.square(action_combinations - x), axis = 1)==0))
            return index[0][0]
        model_index1 = np.apply_along_axis(model_index_fn,  1, exp_his_actions)
        model_index2 = np.apply_along_axis(model_index_fn,  1, take_actions)
        y = np.zeros([samp_size,1])
        for i in range(samp_size):
            temp = qpi_super_lst[idx][model_index1[i]][model_index2[i]].predict(WStorch[i,:].reshape([1,-1]))
            y[i,:] = temp.detach().numpy()
        reg = LinearRegression().fit(ZSA, y)
        proj_alpha_super_lst[idx].append(reg)




    proj_alpha_sz_lst[idx] = []
    ZSAold = np.concatenate((Z, S, Aoldactions), axis = 1)
    for take_action in range(action_space.start, action_space.start + action_space.n):
        # WS = np.concatenate((W, S), axis = 1)
        WStorch = torch.cat((torch.from_numpy(W),torch.from_numpy(S)), dim=1)
        def model_index_fn(x):
            index = np.where((np.sum(np.square(action_combinations - x), axis = 1)==0))
            return index[0][0]
        if Aoldactions.shape[1]==0:
            temp_actions = np.ones((Z.shape[0], 1))* take_action
        else:
            temp_actions = np.concatenate([Aoldactions, np.ones((Z.shape[0], 1))* take_action], axis = 1)
        model_index1 = np.apply_along_axis(model_index_fn,  1, temp_actions)
        y = np.zeros([samp_size,1])
        for i in range(samp_size):
            temp = qpi_dsz_lst[idx][model_index1[i]].predict(WStorch[i,:].reshape([1,-1]))
            y[i,:] = temp.detach().numpy()
        # print(np.min(y))
        reg = LinearRegression().fit(ZSAold, y)
        proj_alpha_sz_lst[idx].append(reg)


    proj_alpha_s_lst[idx] = []
    SAold = np.concatenate((S, Aoldactions), axis = 1)
    for take_action in range(action_space.start, action_space.start + action_space.n):
        # WS = np.concatenate((W, S), axis = 1)
        WStorch = torch.cat((torch.from_numpy(W),torch.from_numpy(S)), dim=1)
        def model_index_fn(x):
            index = np.where((np.sum(np.square(action_combinations - x), axis = 1)==0))
            return index[0][0]
        if Aoldactions.shape[1]==0:
            temp_actions = np.ones((Z.shape[0], 1))* take_action
        else:
            temp_actions = np.concatenate([Aoldactions, np.ones((Z.shape[0], 1))* take_action], axis = 1)
        model_index1 = np.apply_along_axis(model_index_fn,  1, temp_actions)
        y = np.zeros([samp_size,1])
        for i in range(samp_size):
            temp = qpi_ds_lst[idx][model_index1[i]].predict(WStorch[i,:].reshape([1,-1]))
            y[i,:] = temp.detach().numpy()
        # print(np.min(y))
        reg = LinearRegression().fit(SAold, y)
        proj_alpha_s_lst[idx].append(reg)
    


from scipy.special import expit
##### Evaluation ######
def MC_evaluator(env,  policy_name,  config, seed=0, policy_list= None):
    # config = {'episode_num': int, 'verbose': bool}, etc.
    set_seeds(seed)
    
    Rewards = []
    Initial_S = []
    Initial_W = []
    for n in range(config['episode_num']):
        # S = np.zeros([1, 2 * config['episode_num']])
        S = np.empty([1,0])
        # Ahisactions = np.zeros([1, config['episode_num']])
        Ahisactions = np.empty([1,0])
        Aoldactions = np.empty([1,0])
        # Aoldactions = np.zeros([1, config['episode_num']-1])
        try:
            # need seed maybe
            reward = 0
            obs = env.reset() # may need seed
            Initial_S.append(obs['S'])
            Initial_W.append(obs['W'])
            for t in range(env.params['episode_len']):
                # print(t)
                if t==0:
                    Z = np.array([obs['Z']]).reshape([1,-1])
                S = np.append(S, [obs['S']]).reshape([1,-1])
                exp_act = env.exp_act
                Ahisactions = np.append(Ahisactions, [exp_act]).astype(int).reshape([1,-1])
                if policy_name == 'super':
                    if t < config['episode_len']-1:
                        action = policy_super(S, Z, Ahisactions, Aoldactions, proj_alpha_super_lst[t]).astype(int)[0]
                        Aoldactions = np.append(Aoldactions, [action]).astype(int).reshape([1,-1])
                    else:
                        action = policy_super_last(S, Z, Ahisactions, Aoldactions, proj_alpha_super_last_list).astype(int)[0]
                if policy_name == 'sz':
                    action = policy_sz(S, Z, Aoldactions, proj_alpha_sz_lst[t]).astype(int)[0]
                    Aoldactions = np.append(Aoldactions, [action]).astype(int).reshape([1,-1])
                
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



continuousParams = {'episode_len':episode_len, 'offline':False, 
                    'alpha_0':0, 'alpha_a':0.5, 'alpha_s':[0.5, 0.5], 
                    'mu_0':0,    'mu_a':-0.25,  'mu_s':[0.5, 0.5], 
                    'kappa_0':0, 'kappa_a':-0.5, 'kappa_s':[0.5,0.5], 
                    't_0':0,      't_u':tu,     't_s':[-0.5,-0.5]}
ContEnv= ContinuousEnv(continuousParams)
MC_cfg={'episode_num':test_samp_size, 'verbose':False, 'episode_len': episode_len}
val_super,_ , _, _ = MC_evaluator(ContEnv, policy_name = 'super',  config = MC_cfg, seed=0)
# print(val_super)

val_sz,_ , _, _ = MC_evaluator(ContEnv, policy_name = 'sz',  config = MC_cfg, seed=0)
# print(val_sz)




values = np.array([val_sz,  val_super])



print({"common":val_sz, "super":val_super})


