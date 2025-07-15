
# Environment Interface Documentation

This environment follows a standard reinforcement learning (RL) interface for interaction. The agent interacts with the environment through two core methods: `step(action)` and `reset()`. Below is the detailed specification of each method and the values they return.

---

## Interface Overview

### `obs, r, done, info = step(action)`

Performs one step in the environment using the provided `action`.

- **Parameters**
  - `action` (`Any`): The action to take in the environment. The expected type and format depend on the environment's action space.

- **Returns**
  - `obs`(`[0:8]`): The observation (or state) after taking the action. This could be a vector, image, or other structured data.
  - `r` (`float`): The reward received for taking the action. max of obs[4:8] * action[0:4]
  - `done` (`bool`): A flag indicating whether the episode has ended (`True` if terminal).
  - `info` (`dict`): Optional diagnostic information (e.g., environment metrics or debugging data). Does **not** affect learning.

---

### `obs, done, info = reset()`

Resets the environment to an initial state and returns the initial observation.

- **Returns**
  - `obs` (`[0:8]`): The initial observation after resetting the environment.
  - `done`: Always `False` at reset time, indicating the new episode has just begun.
  - `info` (`dict`): Optional initialization metadata or diagnostic data.

---

## Example Usage

```python
obs, done, info = env.reset()

while not done:
    action = policy(obs)
    obs, r, done, info = env.step(action)
```
