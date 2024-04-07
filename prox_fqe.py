#%%
import numpy as np
import torch
from rkhs_torch import ApproxRKHSIV, ApproxRKHSIVCV, _to_tensor
from utils import *
#%%
# option={'n_component':20, 'gamma_f':'auto', 'n_gamma_hs':10, 'n_alphas':20, 'n_components':20, 'cv':5}
#%%
def fit_qpi_cv(S, W, Z, Rewards, Actions, action_space, policy, option, device):
    WS = torch.cat((W,S), dim=1)
    ZS = torch.cat((Z,S), dim=1)
    qpi = []
    #for action in action_space:
    for action in range(action_space.start, action_space.start + action_space.n):
        index = torch.flatten(Actions==action)
        qpi_action = ApproxRKHSIVCV(gamma_f=option['gamma_f'], 
                                    n_gamma_hs=option['n_gamma_hs'],
                                    n_alphas=option['n_alphas'],
                                    cv = option['cv'],
                                    n_components= option['n_components'], 
                                    device = device
        ).fit(
            WS[index,:], 
            Rewards[index],
            ZS[index,:]
        )
        qpi.append(qpi_action.predict)

    def vpi(W, S):
        val_est = torch.zeros((S.shape[0],1), device=S.device)
        k=0
        for action in range(action_space.start, action_space.start + action_space.n):
            val_est = val_est + policy.prob_torch(action, S) * qpi[k](torch.cat((W,S),dim=1))
            k=k+1
        return val_est

    return vpi, qpi

def fit_v0(Episodes, action_space, observation_space, policy, option, device):
    # Init concatenated Episodes data
    Episodes_cat = batch_cat(Episodes, action_space, observation_space, device, verbose=True)
    
    t = len(Episodes[0])-1
    while t >= 0:
        vpi, qpi = fit_qpi_cv(
            S            = Episodes_cat['S'][:,:,t],
            W            = Episodes_cat['W'][:,:,t],
            Z            = Episodes_cat['Z'][:,:,t],
            Rewards      = Episodes_cat['reward'][:,:,t],
            Actions      = Episodes_cat['action'][:,:,t],
            action_space = action_space,
            policy       = policy,
            option       = option,
            device       = device
        )
        if t>0:
            Episodes_cat['reward'][:,:,t-1] = Episodes_cat['reward'][:,:,t-1]\
                + vpi(Episodes_cat['W'][:,:,t], Episodes_cat['S'][:,:,t])
            t = t-1
        else:
            return vpi

