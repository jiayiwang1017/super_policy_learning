import argparse
parser = argparse.ArgumentParser()
# parser.add_argument('--linearity', type =str,  default= 'linear')
parser.add_argument('--n', type =int,  default= 200)
parser.add_argument('--epsilon', type = float, default= 0.5)
parser.add_argument('--seed', type = int, default= 1)
args = parser.parse_args()

# print(args.linearity)
print(args.n)
print(args.epsilon)



from data_generate import prox_data_tabular
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler as Scaler
from sklearn.kernel_approximation import Nystroem


# In[90]:



# In[98]:


from data_generate import prox_data_tabular


# samp_size = 50000

simu = prox_data_tabular(samp_size= args.n, add_noise= True, epsilon = args.epsilon, seed = args.seed)
S = simu.X
A = simu.A
W = simu.W
Z = simu.Z
R = simu.Y
U = simu.U


# In[99]:


#### q function estimation 
### W * S * A 

from sklearn.linear_model import LinearRegression

qlist = np.zeros([2,2,2])
for s in range(2):
    for a in range(2):
        idx = (S== s) & (A == a)
        res = np.zeros([2,1])
        regress = np.zeros([2,2])
        for z in range(2):
            # idxz = (Z == z)
            res[z,0] = np.mean(R[(S== s) & (A == a) & (Z == z)])
            for w in range(2):
                regress[z, w] = np.sum(W[(S== s) & (A == a) & (Z == z)]==w )/ np.sum((S== s) & (A == a) & (Z == z))
        # print(regress)
        # print(res)
        reg = LinearRegression(fit_intercept = False).fit(regress, res)
        qlist[:,s,a] = reg.coef_  



###### projection 
proj_super_list = np.zeros([2,2,2,2]) ### A |  S * Z  * A'
for s in range(2):
    for a in range(2):
        for z in range(2):
            idx = (S==s) & (A==a) & (Z == z)
            for policy_action in range(2):
                proj_super_list[policy_action, s, z, a] = (np.sum(W[idx] == 0) * qlist[0, s, policy_action] + np.sum(W[idx] == 1) * qlist[1, s, policy_action])/np.sum(idx)



proj_SZ_list = np.zeros([2,2,2]) ### A |  S * Z 
for s in range(2):
    for z in range(2):
        idx = (S==s) & (Z == z)
        for policy_action in range(2):
            proj_SZ_list[policy_action, s, z] = (np.sum(W[idx] == 0) * qlist[0, s, policy_action] + np.sum(W[idx] == 1) * qlist[1, s, policy_action])/np.sum(idx)



proj_S_list = np.zeros([2,2]) ### A |  S 
for s in range(2):  
    idx = (S==s)
    for policy_action in range(2):
        proj_S_list[policy_action, s] = (np.sum(W[idx] == 0) * qlist[0, s, policy_action] + np.sum(W[idx] == 1) * qlist[1, s, policy_action])/np.sum(idx)


# In[103]:


##### policy

def policy_super(S, Z, A):
    n = S.shape[0]
    act = np.zeros(n)
    for i in range(n):
        act[i] = np.argmax(proj_super_list[:,S[i],Z[i],A[i]])
    return act


def policy_SZ(S, Z, A):
    n = S.shape[0]
    act = np.zeros(n)
    for i in range(n):
        act[i] = np.argmax(proj_SZ_list[:,S[i],Z[i]])
    return act


def policy_S(S,Z,A):
    n = S.shape[0]
    act = np.zeros(n)
    for i in range(n):
        act[i] = np.argmax(proj_S_list[:,S[i]])
    return act




from sklearn.linear_model import LogisticRegression
X = np.hstack([S.reshape([-1,1]),Z.reshape([-1,1])])
y = A
clf = LogisticRegression(random_state=0).fit(X, y)
clf.predict(X)

def value_fun_est_prop(policy, MC = 500000, seed = 0):
    simu = prox_data_tabular(samp_size = MC, add_noise = False, seed = seed, epsilon = args.epsilon)
    S = simu.X
    Z = simu.Z
    U = simu.U
    A = simu.A
    X = np.hstack([S.reshape([-1,1]),Z.reshape([-1,1])])
    A = clf.predict(X)
    policy_action = policy(S, Z, A)
    rewards =(U - 0.5) * (policy_action - 0.5)
    return np.mean(rewards)


v_s = value_fun_est_prop(policy_S)

v_sz = value_fun_est_prop(policy_SZ)

v_super = value_fun_est_prop(policy_super)


values = np.array([v_s, v_sz, v_super])




values = np.array([v_s, v_sz, v_super])
print({"sonly":v_s, "szonly": v_sz, "super":v_super})
