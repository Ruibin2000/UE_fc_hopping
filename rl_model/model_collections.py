import torch

def epsilon_greedy_prob(obs, E, w, epsilon: float):
    """
    Perform ε-greedy exploration over the power set of binary actions.
    Each bit is independently sampled as 0 or 1 with a mixed probability.

    Args:
        obs (Iterable of float): Observation values, shape [k].
        E   (Iterable of float): Corresponding cost/energy values, same shape.
        w   (float): Scalar weight for cost term.
        epsilon (float): Exploration rate in [0,1].

    Returns:
        p_bits (Tensor of float): A 1D tensor of length k where p_bits[i] is
                                  the probability of setting bit i = 1.
    """
    # 1. Convert inputs to float32 tensors
    obs_t = torch.as_tensor(obs, dtype=torch.float32)
    E_t   = torch.as_tensor(E,   dtype=torch.float32)
    if obs_t.shape != E_t.shape:
        raise ValueError("`obs` and `E` must have the same shape")

    # 2. Compute scores for each action index: score_i = obs_i - w * E_i
    scores = obs_t - w * E_t

    # 3. Identify the greedy (best) action index
    best_idx = int(torch.argmax(scores))

    # 4. Convert that index into its binary vector representation
    #    Determine the number of bits k = length of obs
    k = obs_t.numel()
    #    Format index as binary string padded to length k, then map to int list
    bits = torch.tensor(
        [int(b) for b in format(best_idx, f'0{k}b')],
        dtype=torch.float32
    )

    # 5. Mix the greedy bits with uniform exploration (0.5 per bit)
    #    For each bit i:
    #      if greedy bit = 1: p_i = (1 - ε)*1 + ε*0.5
    #      if greedy bit = 0: p_i = (1 - ε)*0 + ε*0.5
    p_bits = (1 - epsilon) * bits + epsilon * 0.5

    return p_bits, bits

def best_obs_minus_wE_torch(obs, E, w):
    """
    Compute the element-wise score obs_i - w * E_i, then return the index
    and value of the highest-scoring element.

    Args:
        obs (Iterable of numbers): e.g. list, NumPy array, or Torch tensor.
        E   (Iterable of numbers): same shape as obs.
        w   (float or Tensor):     scalar weight to multiply E.

    Returns:
        best_idx   (int):   index of the maximum score in the sequence.
        best_score (float): the maximum score value.
    """
    # 1. Convert inputs to float32 tensors (so we can do tensor operations)
    obs_t = torch.as_tensor(obs, dtype=torch.float32)
    E_t   = torch.as_tensor(E,   dtype=torch.float32)

    # 2. Ensure shapes match
    if obs_t.shape != E_t.shape:
        raise ValueError("`obs` and `E` must have the same shape")

    # 3. Compute scores: score_i = obs_i - w * E_i
    #    This broadcasts w if it's a scalar tensor or Python float.
    scores = obs_t - w * E_t

    # 4. Find the index of the maximum score
    best_idx = int(torch.argmax(scores))

    # 5. Extract the best score as a Python float
    best_score = scores[best_idx].item()
    
    best_obs = obs_t[best_idx].item()

    return best_idx, best_score, best_obs

