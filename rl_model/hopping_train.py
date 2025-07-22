import argparse
import logging
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

from ppo_discrete import PPO
from Env import Env

# Hyperparameters
learning_rate = 0.0005
gamma         = 0.98
lmbda         = 0.95
eps_clip      = 0.1
K_epoch       = 3
T_horizon     = 20
print_interval = 20

# Prepare binary combination utilities
def index_to_binary_list(index: int, length: int) -> list:
    if index >= 2 ** length:
        raise ValueError("Index exceeds the total number of combinations")
    return [int(b) for b in format(index, f'0{length}b')]

def total_combinations(length: int) -> int:
    return 2 ** length

# Training function
def train(env_path: str, rate_path: str, log_dir: str):
    # Setup logging
    logging.basicConfig(filename='train.log', level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s')
    logger = logging.getLogger()

    # TensorBoard writer
    writer = SummaryWriter(log_dir=log_dir)

    env = Env(env_path, rate_path)
    state_dim = 8
    action_dim = total_combinations(4)
    model = PPO(state_dim, action_dim)

    best_score = float('-inf')
    recent_rewards = []  # store recent episode rewards for averaging

    for n_epi in range(1, 10001):
        s = env.reset()[0]
        ep_reward = 0.0
        done = False

        while not done:
            for t in range(T_horizon):
                prob = model.pi(torch.from_numpy(s).float())
                m = Categorical(prob)
                a = m.sample().item()
                a_mapping = index_to_binary_list(a, 4)
                s_prime, r, done, info = env.step(a_mapping)

                model.put_data((s, a, r/100.0, s_prime, prob[a].item(), done))
                s = s_prime
                ep_reward += r / 100.0

                if done:
                    break

            model.train_net()

        # Logging and TensorBoard
        writer.add_scalar('Reward/Episode', ep_reward, n_epi)
        logger.info(f"Episode {n_epi}, Reward {ep_reward:.3f}, Steps {t+1}")
        recent_rewards.append(ep_reward)

        # Save best model
        if ep_reward > best_score:
            best_score = ep_reward
            torch.save(model.state_dict(), 'best_model.pth')
            logger.info(f"New best score {best_score:.3f}, saved best_model.pth")

        # Always save last model
        torch.save(model.state_dict(), 'last_model.pth')

        # Print interval
        if n_epi % print_interval == 0:
            # compute average over the last print_interval episodes
            last_rewards = recent_rewards[-print_interval:]
            avg_reward = sum(last_rewards) / len(last_rewards)
            print(f"# Episode: {n_epi}, Avg Reward: {avg_reward:.3f}")

    writer.close()
    env.close()

# Testing function
def test(env_path: str, rate_path: str, test_episodes: int):
    env = Env(env_path, rate_path)
    state_dim = 8
    action_dim = total_combinations(4)
    model = PPO(state_dim, action_dim)

    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()

    total_reward = 0.0
    for epi in range(1, test_episodes + 1):
        s = env.reset()[0]
        done = False
        ep_reward = 0.0

        while not done:
            prob = model.pi(torch.from_numpy(s).float())
            a = prob.argmax().item()
            a_mapping = index_to_binary_list(a, 4)
            s, r, done, info = env.step(a_mapping)
            ep_reward += r

        print(f"Test Episode {epi}, Reward: {ep_reward:.3f}")
        total_reward += ep_reward

    avg = total_reward / test_episodes
    print(f"Average Test Reward over {test_episodes} episodes: {avg:.3f}")
    env.close()

# Entry point
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', dest='is_testing', help='Run in test mode')
    parser.add_argument('--env_path', type=str, default='train_data/51_s2_2_snr_10mps_seed40_30s.json')
    parser.add_argument('--rate_path', type=str, default='train_data/51_s2_2_rate_10mps_seed40_30s.json')
    parser.add_argument('--log_dir', type=str, default='runs/ppo_experiment')
    parser.add_argument('--test_episodes', type=int, default=10)
    args = parser.parse_args()

    if args.is_testing:
        test(args.env_path, args.rate_path, args.test_episodes)
    else:
        train(args.env_path, args.rate_path, args.log_dir)