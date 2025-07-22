import torch
from torch.distributions import Categorical

class EpsilonBanditPolicy:
    """An ε-bandit policy over a discrete action space with optional exclusion of the all-zero action.

    * Maintains a logit for every action.
    * After each action is taken, call ``update(action_idx, score)`` to add the obtained score to that action's logit.
    * ``probs()`` returns a probability distribution that mixes the softmax of the logits with uniform exploration,
      and can optionally forbid the all-zero action.

    Args:
        action_dim (int): number of possible actions (e.g. 16 for 2^4 binary patterns).
        epsilon (float): exploration rate ∈ [0,1], mixing factor for uniform exploration.
        temperature (float): softmax temperature (>0), lower makes distribution sharper.
        forbid_zero (bool): if True, ensure the all-zero action (index=0) has zero probability.
        device (torch.device | str | None): device for storing logits.
    """

    def __init__(self, action_dim: int, epsilon: float = 0.1, *,
                 temperature: float = 1.0, forbid_zero: bool = False, device=None):
        self.action_dim = action_dim
        self.epsilon = float(epsilon)
        self.temperature = float(temperature)
        self.forbid_zero = bool(forbid_zero)
        self.device = torch.device(device) if device is not None else None

        # Initialize preference logits to zero
        self.logits = torch.zeros(action_dim, dtype=torch.float32,
                                  device=self.device)

    def probs(self) -> torch.Tensor:
        """Compute the mixed ε-softmax probabilities, optionally masking out action 0."""
        # 1. Softmax over logits with temperature
        soft = torch.softmax(self.logits / self.temperature, dim=-1)
        # 2. Uniform distribution
        uniform = torch.full_like(soft, 1.0 / self.action_dim)
        # 3. ε-mix: mostly exploit soft, partly explore uniform
        mixed = (1.0 - self.epsilon) * soft + self.epsilon * uniform

        if self.forbid_zero:
            # Zero out probability of the all-zero action (index=0)
            mixed = mixed.clone()
            mixed[0] = 0.0
            # Renormalize so sum = 1
            total = mixed.sum()
            if total > 0:
                mixed /= total
            else:
                # Fallback to uniform over non-zero actions
                mixed = torch.full((self.action_dim,), 1.0 / (self.action_dim - 1),
                                    device=mixed.device)
                mixed[0] = 0.0
        return mixed

    def sample_action(self) -> int:
        """Sample an action index according to the current policy distribution."""
        dist = Categorical(self.probs())
        return int(dist.sample().item())

    def update(self, action_idx: int, score: float):
        """Increment the logit of `action_idx` by the given score."""
        if not 0 <= action_idx < self.action_dim:
            raise IndexError("action_idx out of bounds")
        self.logits[action_idx] += float(score)

    def reset(self):
        """Reset all logits back to zero."""
        self.logits.zero_()
        return self

    def to(self, device):
        """Move logits to a specified device."""
        self.device = torch.device(device)
        self.logits = self.logits.to(self.device)
        return self

    __call__ = probs  # allow direct call to get probabilities

# Example usage
if __name__ == "__main__":
    policy = EpsilonBanditPolicy(action_dim=16, epsilon=0.2,
                                  temperature=1.0, forbid_zero=True)
    for step in range(5):
        a = policy.sample_action()
        print(f"Sampled action: {a}")
        score = float(torch.randn(()))  # sample random score
        policy.update(a, score)
        print(f"Updated logits: {policy.logits}")
        print(f"Current policy probs: {policy.probs().tolist()}\n")