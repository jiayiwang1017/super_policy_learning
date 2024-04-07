

import argparse
parser = argparse.ArgumentParser()
# parser.add_argument('--linearity', type =str,  default= 'linear')
parser.add_argument('--n', type =int,  default= 200)
parser.add_argument('--epsilon', type = float, default= 0.5)
parser.add_argument('--seed', type = int, default= 1)
args = parser.parse_args()

print(args.n)
print(args.epsilon)



from data_generate import prox_data_simple3
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# In[83]:


def value_fun(policy, MC = 50000, seed = 0):
    simu = prox_data_simple3(samp_size = MC, add_noise = False, seed = seed, epsilon =args.epsilon)
    data = simu.data
    S = data[["X"]].values
    Z = data[["Z"]].values
    U = data[["U"]].values
    A = data[["A"]].values
    policy_action = policy(S, Z, A)
    rewards =  (U)   *  (policy_action.reshape(U.shape) - 0.5) 
    return np.mean(rewards)


# In[84]:


simu = prox_data_simple3(samp_size = args.n, add_noise = True,  epsilon = args.epsilon, seed = args.seed)
train= simu.data
samp_size = args.n



# import prox_fqe
from prox_fqe import fit_qpi_cv
A = train[["A"]]
S = train[["X"]]
Z = train[["Z"]]
W = train[["W"]]
Y = train[["Y"]]


# In[90]:


import numpy as np
from rkhs_torch import ApproxRKHSIV, ApproxRKHSIVCV, _to_tensor
from utils import *
#%%
# option={'n_component':20, 'gamma_f':'auto', 'n_gamma_hs':10, 'n_alphas':20, 'n_components':20, 'cv':5}
#%%
def fit_qpi_cv(S, W, Z, Rewards, Actions, action_space,option, device):
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



from gym import spaces
action_space = spaces.Discrete(2)


import numpy as np
from scipy.special import expit


option={'n_component':20, 'gamma_f':'auto', 'n_gamma_hs':10, 'n_alphas':20, 'n_components':20, 'cv':5}

Wtorch = torch.tensor(W.values)
Storch = torch.tensor(S.values)
Ztorch = torch.tensor(Z.values)
Ytorch = torch.tensor(Y.values)
Atorch = torch.tensor(A.values)

qpi = fit_qpi_cv(S = Storch, W = Wtorch, Z = Ztorch, Rewards = Ytorch, Actions = Atorch, action_space = action_space,option = option, device = 'cpu')


#### linear projections
from sklearn.linear_model import LinearRegression

ZStorch = torch.cat((Ztorch,Storch), dim=1)
WStorch = torch.cat((Wtorch,Storch), dim=1)
ZSAtorch = torch.cat((Ztorch, Storch, Atorch), dim = 1)
proj_alpha_list = []
proj_alpha_list_sz = []
proj_alpha_list_s = []
for take_action in range( action_space.n):
    y = qpi[take_action].predict(WStorch)
    reg = LinearRegression().fit(ZSAtorch.detach().numpy(), y.detach().numpy())
    proj_alpha_list.append(reg)
    reg = LinearRegression().fit(ZStorch.detach().numpy(), y.detach().numpy())
    proj_alpha_list_sz.append(reg)
    reg = LinearRegression().fit(Storch.detach().numpy(), y.detach().numpy())
    proj_alpha_list_s.append(reg)


# In[95]:


### super_policy 
def policy_super(S, Z, A):
    Storch = torch.tensor(S)
    Ztorch = torch.tensor(Z)
    Atorch = torch.tensor(A)
    super_act = np.zeros(Z.shape[0])
    projq = np.zeros([Storch.shape[0] ,  action_space.n])
    for take_action in range(action_space.n):
        ZSAtorch = torch.cat((Ztorch, Storch, Atorch), dim=1)
        projq[:,take_action] = proj_alpha_list[take_action].predict(ZSAtorch.detach().numpy()).reshape(-1)
    super_act = np.argmax(projq,axis = 1)
    return super_act


### sz_policy 
def policy_sz(S, Z, A):
    Storch = torch.tensor(S)
    Ztorch = torch.tensor(Z)
    Atorch = torch.tensor(A)
    super_act = np.zeros(Z.shape[0])
    projq = np.zeros([Storch.shape[0] ,action_space.n])
    for take_action in range( action_space.n):
        ZStorch = torch.cat((Ztorch, Storch), dim=1)
        projq[:,take_action] = proj_alpha_list_sz[take_action].predict(ZStorch.detach().numpy()).reshape(-1)
    super_act = np.argmax(projq,axis = 1)
    return super_act



### s_policy 
def policy_s(S, Z, A):
    Storch = torch.tensor(S)
    Ztorch = torch.tensor(Z)
    Atorch = torch.tensor(A)
    super_act = np.zeros(Z.shape[0])
    projq = np.zeros([Storch.shape[0] , action_space.n])
    for take_action in range(action_space.n):
        # ZSAtorch = torch.cat((Ztorch, Storch), dim=1)
        projq[:,take_action] = proj_alpha_list_s[take_action].predict(Storch.detach().numpy()).reshape(-1)
    super_act = np.argmax(projq,axis = 1)
    return super_act



# In[97]:


v_s = value_fun(policy_s, MC=50000, seed = args.seed)
v_sz = value_fun(policy_sz, MC=50000, seed = args.seed)
v_super = value_fun(policy_super, MC=50000, seed = args.seed)

# print(v_s, v_sz, v_super)



values = np.array([v_s, v_sz, v_super])
print({"sonly":v_s, "szonly": v_sz, "super":v_super})



