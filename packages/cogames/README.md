# CoGames: Cogs vs Clips Multi-Agent RL Environment

CoGames is a collection of multi-agent cooperative and competitive environments designed for reinforcement learning
research.

## The game: Cogs vs Clips

Multiple "Cog" agents, controlled by user-provided policies, must cooperate to extract Hearts from the environment.
Doing so requires gathering resources, operating machinery, and assembling components. Many steps will require
interacting with a "station". Many such interactions will require multiple cogs working in tandem.

Your Cogs' efforts may be thwarted by Clips: NPC agents that disable stations or otherwise impede progress.

<p align="middle">
<img src="assets/showoff.gif" alt="Example Cogs vs Clips video">
<br>

There are many game configurations available, with different map sizes, resource and station layouts, and game rules.
Overall, Cogs vs Clips aims to present rich environments with:

- **Resource management**: Energy, materials (carbon, oxygen, germanium, silicon), and crafted components
- **Station-based interactions**: Different stations provide unique capabilities (extractors, assemblers, chargers,
  chests)
- **Sparse rewards**: Agents receive rewards only upon successfully crafting target items (hearts)
- **Partial observability**: Agents have limited visibility of the environment
- **Required multi-agent cooperation**: Agents must coordinate to efficiently use shared resources and stations

Cogs should refer to their [MISSION.md](MISSION.md) for a thorough description of the game mechanics.

## Quick Start

```bash
# Install
uv pip install cogames

# List games
cogames games

# Play an episode of the machina_1 game
cogames play machina_1 --interactive

# Train a policy
cogames train machina_1 --policy simple --steps 10000

# Watch or play along side your trained policy
cogames play machina_1 --interactive --policy simple --policy-data ./train_dir/policy.pt

# Evaluate your policy
cogames evaluate machina_1 simple:./train_dir/policy.pt
```

## Commands

### `cogames games [game]`

List all games or describe a specific game.

**Options:**

- `--save PATH`: Save game config to YAML/JSON file

### `cogames play [game]`

Play an episode of the specified game. Cogs' actions are determined by the provided policy.

**Options:**

- `--policy PATH`: Policy class (default: random)
- `--policy-data PATH`: Path to weights file/dir
- `--steps N`: Number of steps (default: 1000)
- `--render MODE`: 'gui' or 'text' (default: gui)
- `--interactive`: Interactive mode (default: true)

`cogames play` supports a gui-based and text-based game renderer, both of which support many features to inspect agents
and manually play alongside them.

### `cogames train [game]`

Train a policy on a game.

**Options:**

- `--policy PATH`: Policy class (default: SimplePolicy)
- `--initial-weights PATH`: Starting weights
- `--checkpoints PATH`: Save location (default: ./train_dir)
- `--steps N`: Training steps (default: 10000)
- `--device STR`: 'auto', 'cpu', or 'cuda' (default: auto)
- `--batch-size N`: Batch size (default: 4096)
- `--num-workers N`: Worker processes (default: CPU count)

### Custom Policy Architectures

To get started, `cogames` supports some torch-nn-based policy architectures out of the box (such as SimplePolicy). To
supply your own, you will want to extend `cogames.policy.Policy`.

```python
from cogames.policy import Policy

class MyPolicy(Policy):
    def __init__(self, observation_space, action_space):
        self.network = MyNetwork(observation_space, action_space)

    def get_action(self, observation, agent_id=None):
        return self.network(observation)

    def reset(self):
        pass

    def save(self, path):
        torch.save(self.network.state_dict(), path)

    @classmethod
    def load(cls, path, env=None):
        policy = cls(env.observation_space, env.action_space)
        policy.network.load_state_dict(torch.load(path))
        return policy
```

To train with using your class, supply a path to it with the `--policy`argument, e.g.
`cogames train --policy path.to.MyPolicy`.

#### Environment API

The underlying environment follows the Gymnasium API:

```python
from cogames import get_game
from mettagrid.envs import MettaGridEnv

# Load a game configuration
config = get_game("assembler_2_complex")

# Create environment
env = MettaGridEnv(env_cfg=config)

# Reset environment
obs, info = env.reset()

# Game loop
for step in range(1000):
    # Your policy computes actions for all agents
    actions = policy.get_actions(obs)  # Dict[agent_id, action]

    # Step environment
    obs, rewards, terminated, truncated, info = env.step(actions)

    if terminated or truncated:
        obs, info = env.reset()
```

### `cogames evaluate [game] [policies...]`

Evaluate one or more policies.

**Policy spec format:** `{class_path}[:data_path][:proportion]`

**Examples:**

```bash
# Single policy
cogames evaluate machina_1 random

# Trained policy
cogames evaluate machina_1 simple:./train_dir/model.pt

# Latest checkpoint in directory
cogames evaluate machina_1 simple:./train_dir/

# Mixed population (1:2 ratio)
cogames evaluate machina_1 simple:./model.pt:1 random::2
```

**Options:**

- `--episodes N`: Number of episodes (default: 10)
- `--action-timeout-ms N`: Timeout per action (default: 250ms)

When multiple policies are provided, `cogames evaluate` fixes the number of agents each policy will control, but
randomizes their assignments each episode.

### `cogames make-game [base_game]`

Create custom game configuration.

**Options:**

- `--agents N`: Number of agents (default: 2)
- `--width W`: Map width (default: 10)
- `--height H`: Map height (default: 10)
- `--output PATH`: Save to file

You will be able to provide your specified `--output` path as the `game` argument to other `cogames` commmands.

### `cogames version`

Show version info for mettagrid, pufferlib-core, and cogames.

## Citation

If you use CoGames in your research, please cite:

```bibtex
@software{cogames2024,
  title={CoGames: Multi-Agent Cooperative Game Environments},
  author={Metta AI},
  year={2024},
  url={https://github.com/metta-ai/metta}
}
```
