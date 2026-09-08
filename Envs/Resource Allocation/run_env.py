import h5py
import numpy as np
import os
import pickle
from env import Env
from argparser import args


class Greedy(object):
    def __init__(self, act_size, n_servers, name='greedy'):
        self.name = name     
        self.n_servers = n_servers
        self.act_size = act_size

    def step(self, obs):
        m_pending = obs[20:30]
        #find the max number of m_cpu and its index
        m_pending = (m_pending.min(), np.where(m_pending == m_pending.min())[0])
        return np.random.choice(m_pending[1], )
    

def main():
    env = Env()
    model = Greedy(act_size=args.n_servers, n_servers=args.n_servers)     
    
    env_name = model.name
    done = False
    obs = env.reset()
    action = 0

    # Warm-up
    for _ in range(args.warmup):
        action = (action + 1) % args.n_servers
        obs, _, _, _ = env.step(action)
    
    for _ in range(args.n_tasks):
        action = int(model.step(obs))
        new_obs, reward, done, _ = env.step(action)

if __name__ == '__main__':
    main()