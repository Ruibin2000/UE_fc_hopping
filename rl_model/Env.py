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
        self.horizon = len(self.rate_array)-1
        self.reset()
        max_per_antenna = self.rate_array.max(axis=0)
        # self.energy_consume_array = np.array([10, 10, 10, 10]) 
        self.energy_consume_array = 0.3 * max_per_antenna  # Example energy consumption per antenna
        print(f"Energy consumption per antenna: {self.energy_consume_array}")

    def reset(self):
        self.index = 0
        self.obs = self.step_true()[0]
        self.done = False
        self.info = []
        return self.obs, self.done, self.info
        
    def step(self, action_list):
        action = np.array(action_list)
        self.obs = np.r_[self.rate_array[self.index] * action, self.snr_array[self.index] * action]
        self.r = np.max(self.rate_array[self.index] * action) - np.sum(self.energy_consume_array * action)

        if self.index >= self.horizon - 1:
            self.done = True
        else:
            self.index = self.index + 1

        return self.obs, self.r, self.done, self.info
    
    def step_true(self):
        self.state = np.r_[self.rate_array[self.index] * np.ones(4), self.snr_array[self.index] * np.ones(4)]
        self.next_state = np.r_[self.rate_array[self.index + 1] * np.ones(4), self.snr_array[self.index + 1] * np.ones(4)]

        return self.state, self.next_state
    
    def max_rate(self, action_list):
        action = np.array(action_list)
        return np.max(self.rate_array[self.index] * action)


def main():
    env = Env('train_data/51_s2_2_snr_10mps_seed40_30s.json', 'train_data/51_s2_2_rate_10mps_seed40_30s.json')
    
    obs, done, info = env.reset()

    while not done:
        action = [1,1,1,1]
        obs, r, done, info = env.step(action)
        print(r)

if __name__ == '__main__':
    main()