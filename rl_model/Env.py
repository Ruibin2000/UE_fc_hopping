import json
import numpy as np

class Env:
    def __init__(self, file_snr, file_rate):
        with open(file_snr, 'r') as f:
            self.snr_list = json.load(f)
            self.snr_array = np.array(self.snr_list)

        with open(file_rate, 'r') as f:
            self.rate_list = json.load(f)
            self.rate_array = np.array(self.rate_list)

        self.reset()
        self.energy_consume_array = np.array([10, 10, 10, 10])

    def reset(self):
        self.obs = np.zeros(8)
        self.done = False
        self.info = []
        self.index = 0
        return self.obs, self.done, self.info
        
    def step(self, action_list):
        action = np.array(action_list)
        self.obs = np.r_[self.rate_array[self.index] * action, self.snr_array[self.index] * action]
        self.r = np.max(self.rate_array[self.index] * action) - np.sum(self.energy_consume_array * action)

        if self.index >= len(self.rate_array)-1:
            self.done = True
        else:
            self.index = self.index + 1

        return self.obs, self.r, self.done, self.info


def main():
    env = Env('train_data/51_s2_2_snr_10mps_seed40_30s.json', 'train_data/51_s2_2_rate_10mps_seed40_30s.json')
    
    obs, done, info = env.reset()

    while not done:
        action = [1,1,1,1]
        obs, r, done, info = env.step(action)
        print(r)

if __name__ == '__main__':
    main()