from ppo_discrete import PPO
from Env import Env

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

#Hyperparameters
learning_rate = 0.0005
gamma         = 0.98
lmbda         = 0.95
eps_clip      = 0.1
K_epoch       = 3
T_horizon     = 20

def index_to_binary_list(index: int, length: int) -> list:
    if index >= 2**length:
        raise ValueError("Index exceeds the total number of combinations")
    return [int(b) for b in format(index, f'0{length}b')]

def total_combinations(length: int) -> int:
    return 2 ** length

def train():
    env = Env('train_data/51_s2_2_snr_10mps_seed40_30s.json', 'train_data/51_s2_2_rate_10mps_seed40_30s.json')

    state_dim = 8
    action_dim = total_combinations(4)
    model = PPO(state_dim, action_dim)
    score = 0.0
    epi_len = []
    print_interval = 20

    for n_epi in range(10000):
        s = env.reset()[0]
        done = False
        while not done:
            for t in range(T_horizon):
                prob = model.pi(torch.from_numpy(s).float())
                m = Categorical(prob)
                a = m.sample().item()
                a_mapping = index_to_binary_list(a,4)
                s_prime, r, done, info = env.step(a_mapping)
                # print(r)
                # env.render()
                model.put_data((s, a, r/100.0, s_prime, prob[a].item(), done))

                s = s_prime

                score += r / 100
                if done:
                    break

            model.train_net()
        epi_len.append(t)
        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.3f}, avg epi length :{}".format(n_epi, score/print_interval, int(np.mean(epi_len))))
            score = 0.0
            epi_len = []

    env.close()


if __name__ == '__main__':
    train()