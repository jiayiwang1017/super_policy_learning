#%% Attach gym
MODULE_PATH = "./gym/gym/__init__.py"
MODULE_NAME = "gym"
import importlib
import sys
from typing import OrderedDict
spec = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module 
spec.loader.exec_module(module)

#%%
from collections import OrderedDict

import numpy as np
import scipy
from scipy.special import expit
from scipy.stats import multivariate_normal
from scipy.stats import uniform
from scipy.stats import randint

import gym
from gym import spaces
from gym.utils import seeding


class TabularEnv(gym.Env):
    def __init__(self, params, device='cpu'):
        self.name = "Tabular Env"
        self.device=device

        self.params=params #{'episode_len':5, 'offline':True/False,
                           # 'alpha_0', 'alpha_a', 'alpha_s', 
                           # 'mu_0',    'mu_a',    'mu_s', 
                           # 'kappa_0', 'kappa_a', 'kappa_s', 
                           # 't_0',     't_u',     't_s'}
        ########### Need to check assumptions for above params ########

        self.action_space = spaces.Discrete(2) # {0, 1}
        self.observation_space = spaces.Dict({
            'O': spaces.Discrete(2), 
            'U': spaces.Discrete(2), 
        })
        self.epsilon = params['epsilon']

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.obs_last = self.observation_space.sample()
        U = self.obs_last['U']
        # Create pseudo action A in {-1,1} from S to generate Z,U,W
        probA = self.epsilon * 1 *  (U == 1 ) + (1 - self.epsilon ) * 1 *  (U == 0 ) 
        probO = 0.7 * 1 *  (U == 1 ) + (1 - 0.7 ) * 1 *  (U == 0 )
        A = (np.random.uniform(size = 1) - probA < 0).astype(int)
        O = (np.random.uniform(size = 1) - probO < 0).astype(int)
        self.obs_last['U'] = U
        self.obs_last['O'] = O
        self.exp_act = int((A + 1)/2)
        if self.params['offline']:
            self.act_last = int((A + 1)/2)

        self.n_step=0
        self.his = []
        return self.obs_last

    def step(self, action):
        # Based on U_t, S_t, W_t, Z_t in self.obs_last and action (A_t) to generate reward R_t
        # and update self.obs_last to U_{t+1}, S_{t+1}, W_{t+1}, Z_{t+1}, self.act_last=action
        if not self.action_space.contains(action):
            raise "invalid action"

        U = self.obs_last['U']
        O = self.obs_last['O']
        self.his.append([O[0],self.exp_act])
        if self.params['offline']:
            action = self.act_last
        # reward = expit((action - 0.5) * (U + S[0] - 2*S[1])) + (np.random.rand()-0.5)/10.
        reward = 1 * ((U - 0.5 )   *  (action - 0.5) > 0 )
        probUnext =  1 *  (U != action ) + 0 * 1 *  (U == action) 
        U = (np.random.uniform(size = 1) - probUnext < 0).astype(int)
        probA = self.epsilon * 1 *  (U == 1 ) + (1 - self.epsilon ) * 1 *  (U == 0 ) 
        probO = 0.7 * 1 *  (U == 1 ) + (1 - 0.7 ) * 1 *  (U == 0 )
        A = (np.random.uniform(size = 1) - probA < 0).astype(int)
        O = (np.random.uniform(size = 1) - probO < 0).astype(int)

        self.obs_last = OrderedDict([('U',U), ('O',O)])
        self.exp_act = int((A + 1)/2)
        if self.params['offline']:
            self.act_last = int((A+1)/2)
        else:
            self.act_last = action

        if self.n_step >= self.params['episode_len']:
            done = True
        else:
            done = False
            self.n_step += 1

        return self.obs_last, reward, done, {}















#%%
class ContinuousEnv(gym.Env):
    def __init__(self, params, device='cpu'):
        self.name = "Continuous Env"
        self.device=device

        self.params=params #{'episode_len':5, 'offline':True/False,
                           # 'alpha_0', 'alpha_a', 'alpha_s', 
                           # 'mu_0',    'mu_a',    'mu_s', 
                           # 'kappa_0', 'kappa_a', 'kappa_s', 
                           # 't_0',     't_u',     't_s'}
        self.COV = [[1,0.25,0.5], [0.25,1,0.5], [0.5,0.5,1]]
        self.alpha_0 = params['alpha_0']
        self.alpha_a = params['alpha_a']
        self.alpha_s = np.array(params['alpha_s']) # 2-vector
        self.mu_0    = params['mu_0']
        self.mu_a    = params['mu_a']
        self.mu_s    = np.array(params['mu_s']) # 2-vector
        self.kappa_0 = params['kappa_0']
        self.kappa_a = params['kappa_a']
        self.kappa_s = np.array(params['kappa_s']) # 2-vector
        self.t_0     = params['t_0']
        self.t_u     = params['t_u']
        self.t_s     = np.array(params['t_s']) # 2-vector
        ########### Need to check assumptions for above params ########

        self.action_space = spaces.Discrete(2) # {0, 1}
        self.observation_space = spaces.Dict({
            'S': spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
            'U': spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            'W': spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            'Z': spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        })

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.obs_last = self.observation_space.sample()
        S = self.obs_last['S']
        # Create pseudo action A in {-1,1} from S to generate Z,U,W
        A = -1 + 2*int(np.random.rand() < expit(-(self.t_0 + self.t_u*self.kappa_0 + (self.t_s+self.t_u*self.kappa_s) @ S)))
        ZWU = multivariate_normal.rvs(size=1, 
                mean = np.array([self.alpha_0 + self.alpha_a*A + self.alpha_s@S,
                                 self.mu_0    + self.mu_a*A    + self.mu_s@S,
                                 self.kappa_0 + self.kappa_a*A + self.kappa_s@S]),
                cov = self.COV, 
                random_state=self.seed()[0]%(2**32))
        self.obs_last['Z'] = ZWU[0]
        self.obs_last['W'] = ZWU[1]
        self.obs_last['U'] = ZWU[2]
        self.exp_act = int((A + 1)/2)
        if self.params['offline']:
            self.act_last = int((A + 1)/2)

        self.n_step=0
        return self.obs_last

    def step(self, action):
        # Based on U_t, S_t, W_t, Z_t in self.obs_last and action (A_t) to generate reward R_t
        # and update self.obs_last to U_{t+1}, S_{t+1}, W_{t+1}, Z_{t+1}, self.act_last=action
        if not self.action_space.contains(action):
            raise "invalid action"

        U = self.obs_last['U']
        S = self.obs_last['S']
        if self.params['offline']:
            action = self.act_last
        # reward = expit((action - 0.5) * (U + S[0] - 2*S[1])) + (np.random.rand()-0.5)/10.
        reward = expit((action - 0.5) * (U)) + (np.random.rand()-0.5)/10.

        # Use previous (U,S) and current action to generate current S
        S = multivariate_normal.rvs(size=1, mean= S + 2*(action-0.5)*U*np.ones_like(S), random_state=self.seed()[0]%(2**32))
        # Create pseudo action A in {-1,1} from S to generate Z,U,W
        A = -1 + 2*int(np.random.rand() < expit(-(self.t_0 + self.t_u*self.kappa_0 + (self.t_s+self.t_u*self.kappa_s) @ S)))
        ZWU = multivariate_normal.rvs(size=1, 
                mean = np.array([self.alpha_0 + self.alpha_a*A + self.alpha_s@S,
                                 self.mu_0    + self.mu_a*A    + self.mu_s@S,
                                 self.kappa_0 + self.kappa_a*A + self.kappa_s@S]),
                cov = self.COV, 
                random_state=self.seed()[0]%(2**32))

        self.obs_last = OrderedDict([('S',S), ('U',ZWU[2]), ('W',ZWU[1]), ('Z',ZWU[0])])
        self.exp_act = int((A + 1)/2)
        if self.params['offline']:
            self.act_last = int((A+1)/2)
        else:
            self.act_last = action

        if self.n_step >= self.params['episode_len']:
            done = True
        else:
            done = False
            self.n_step += 1

        return self.obs_last, reward, done, {}

