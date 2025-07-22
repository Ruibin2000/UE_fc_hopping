import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.distributions import Bernoulli

from model_collections import best_obs_minus_wE_torch, epsilon_greedy_prob
from Env import Env

# ---------------------------
# Utility helpers
# ---------------------------

def index_to_list(index: int, length: int) -> list[int]:
    """Return a binary list with 1 at *index* and 0 elsewhere."""
    if index >= length:
        raise ValueError("Index exceeds the total number of antennas")
    return [1 if i == index else 0 for i in range(length)]

def sample_bits_nonzero(probs: torch.Tensor, max_attempts: int = 10) -> torch.Tensor:
    """
    Sample a 0/1 vector from Bernoulli(probs) but ensure at least one bit = 1.
    - Tries up to max_attempts times.
    - If still all-zero, force one random bit to 1.
    """
    dist = Bernoulli(probs)
    for _ in range(max_attempts):
        bits = dist.sample().int()
        if bits.sum() > 0:
            return bits
    # fallback: pick one index at random and set it to 1
    idx = torch.randint(len(probs), (1,)).item()
    bits = torch.zeros_like(probs, dtype=torch.int)
    bits[idx] = 1
    return bits

# ---------------------------
# Testing + live logging
# ---------------------------

def test(env_path: str, rate_path: str, test_episodes: int, save_dir: str | None = None):
    """Run ε-greedy evaluation and generate two diagnostic plots.

    1. best_score vs. ε-greedy reward (per step timeline)
    2. The four rate components (next_state[:4]) per step, along with
       max_rate_epsilon and max_rate_true (highlighted).
    """
    env = Env(env_path, rate_path)
    E = env.energy_consume_array  # Assumed constant over episode

    # --- buffers for plotting ---
    t_all: list[int] = []
    best_scores: list[float] = []
    rewards: list[float] = []

    rate_components = [[], [], [], []]  # 4 separate lists
    max_rate_eps: list[float] = []
    max_rate_true: list[float] = []

    global_step = 0  # timeline across ALL episodes

    for epi in range(1, test_episodes + 1):
        _ = env.reset()
        # The env in this project seems to expose a helper to retrieve current states:
        state, next_state = env.step_true()  # ground-truth transition (no cost)
        done = False

        while not done:
            # Greedy best (oracle) action for current *next_state*
            best_idx, best_score, best_obs = best_obs_minus_wE_torch(obs=next_state[:4], E=E, w=1.0)
            best_action = index_to_list(best_idx, 4)

            # ε-greedy probabilities derived from current *state*
            probs, base_probs = epsilon_greedy_prob(obs=state[:4], E=E, w=1.0, epsilon=0.2)
            a_bits  = sample_bits_nonzero(probs).tolist()

            # Step using ε-greedy action
            state, r, done, info = env.step(a_bits)

            # Compute diagnostic quantities
            max_rate_eps_step = env.max_rate(a_bits)
            max_rate_true_step = env.max_rate(best_action)

            # --- log for plotting ---
            t_all.append(global_step)
            best_scores.append(best_score)
            rewards.append(r)

            # Store each of the 4 rate components from next_state (before step)
            for i in range(4):
                rate_components[i].append(next_state[i])
            max_rate_eps.append(max_rate_eps_step)
            max_rate_true.append(max_rate_true_step)

            # Advance timeline
            global_step += 1

            # Prepare for next loop iteration
            if not done:
                _, next_state = env.step_true()
            else:
                next_state = state  # final state


    # ---------------------------
    # Plot 1: best_score vs reward
    # ---------------------------
    plt.figure(figsize=(10, 4))
    plt.plot(t_all, best_scores, label="best_score (oracle)")
    plt.plot(t_all, rewards,      label="ε-greedy reward", linestyle="--")
    plt.xlabel("Time step")
    plt.ylabel("Score / Reward")
    plt.title("Oracle best_score vs ε-greedy reward")
    plt.legend()
    plt.tight_layout()
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(Path(save_dir) / "score_vs_reward.png", dpi=150)
    plt.show()

    # ---------------------------
    # Plot 2: rate components + max rates
    # ---------------------------
    plt.figure(figsize=(10, 5))
    # four individual components (thin lines)
    comp_labels = ["rate_0", "rate_1", "rate_2", "rate_3"]
    for i in range(4):
        plt.plot(t_all, rate_components[i], alpha=0.3, label=comp_labels[i])

    # Highlighted lines for max_rate_eps and max_rate_true
    plt.plot(t_all, max_rate_eps,  linewidth=2.5, label="max_rate_ε-greedy", alpha = 0.8)
    plt.plot(t_all, max_rate_true, linewidth=2.5, linestyle="--", alpha = 0.8, label="max_rate_oracle")

    plt.xlabel("Time step")
    plt.ylabel("Rate")
    plt.title("Per-step rate components and max rates")
    plt.legend()
    plt.tight_layout()
    if save_dir:
        plt.savefig(Path(save_dir) / "rates_components.png", dpi=150)
    plt.show()

    print("Finished testing and plotting.")

# ---------------------------
# CLI entry
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_path",  type=str, default="train_data/51_s2_2_snr_10mps_seed40_30s.json")
    parser.add_argument("--rate_path", type=str, default="train_data/51_s2_2_rate_10mps_seed40_30s.json")
    parser.add_argument("--episodes",   type=int, default=1, help="Number of test episodes")
    parser.add_argument("--save_dir",   type=str, default='runs/epsilon_experiment', help="Optional directory to save plots")
    args = parser.parse_args()

    # Optional: basic logging to console
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    test(args.env_path, args.rate_path, args.episodes, args.save_dir)