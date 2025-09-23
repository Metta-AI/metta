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

There are many mission configurations available, with different map sizes, resource and station layouts, and game rules.
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

# List missions
cogames missions

# Play an episode of the machina_1 game
cogames play training_facility_1 --interactive

# Train a policy in a simple, single-agent game
cogames train training_facility_1 --policy simple

# Watch or play along side your trained policy
cogames play training_facility_1 --policy simple --policy-data ./train_dir/policy.pt --interactive

# Evaluate your policy
cogames evaluate training_facility_1 --policy simple --policy-data ./train_dir/policy.pt
```

## Commands

### `cogames missions [mission_name]`

Lists all missions and their high-level specs.

If a `mission_name` is provided, it describe a specific mission in detail.

### `cogames play [mission]`

Play an episode of the specified mission. Cogs' actions are determined by the provided policy.

**Options:**

- `--policy PATH`: Policy class (default: random)
- `--policy-data PATH`: Path to weights file/dir
- `--steps N`: Number of steps (default: 1000)
- `--render MODE`: 'gui' or 'text' (default: gui)
- `--interactive`: Interactive mode (default: true)

`cogames play` supports a gui-based and text-based game renderer, both of which support many features to inspect agents
and manually play alongside them.

### `cogames train [mission]`

Train a policy on a mission.

**Options:**

- `--policy PATH`: Policy class (default: SimplePolicy)
- `--initial-weights PATH`: Starting weights
- `--checkpoints PATH`: Save location (default: ./train_dir)
- `--steps N`: Training steps (default: 10000)
- `--device STR`: 'auto', 'cpu', or 'cuda' (default: auto)
- `--batch-size N`: Batch size (default: 4096)
- `--num-workers N`: Worker processes (default: CPU count)

### `cogames eval [game] [policies...]`

To specify policies to evaluate, you can either provide `--policy` and `--policy-data` arguments as seen in other `cogames` commands, or can provide a list of policy specs:
**Policy spec format:** `{class_path}[:data_path][:proportion]`

```bash
# Trained policy
cogames eval machina_1 --policy simple --policy-data train_dir/model.pt

# Or, equivalently
cogames eval machina_1 simple:train_dir/model.pt

# Mixed population of agents, 3/8 of which steered by your policy, the rest by a random-action policy
cogames eval machina_1 simple:train_dir/model.pt:3 random::5
```

When multiple policies are provided, `cogames eval` fixes the number of agents each policy will control, but
randomizes their assignments each episode.

### Custom Policy Architectures

To get started, `cogames` supports some torch-nn-based policy architectures out of the box (such as SimplePolicy). To
supply your own, you will want to extend `cogames.policy.Policy`.

```python
from cogames.policy import Policy

class MyPolicy(Policy):
    def __init__(self, observation_space, action_space):
        self.network = MyNetwork(observation_space, action_space)
