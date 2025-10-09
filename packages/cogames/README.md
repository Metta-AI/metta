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
cogames train training_facility_1 simple

# Watch or play along side your trained policy
cogames play training_facility_1 simple:train_dir/policy.pt --interactive

# Evaluate your policy
cogames eval training_facility_1 simple:./train_dir/policy.pt
```

## Commands

Most commands are of the form `cogames <command> [MISSION] [POLICY] [OPTIONS]`

To specify a `MISSION`, you can:
- Use a mission name from the default registry emitted by `cogames missions`, e.g. `training_facility_1`
- Use a path to a mission configuration file, e.g. path/to/mission.yaml"

To specify a `POLICY`, provide an argument with up to three parts `CLASS[:DATA][:PROPORTION]`:
- `CLASS`: Policy shorthand (`simple`, `random`, `lstm`, ...) or fully qualified class path like `cogames.policy.random.RandomPolicy`.
- `DATA`: Optional path to a weights file or directory. When omitted, defaults to the policy's built-in weights.
- `PROPORTION`: Optional positive float specifying the relative share of agents that use this policy (default: 1.0).

### `cogames missions [MISSION]`

Lists all missions and their high-level specs.

If a mission is provided, it describe a specific mission in detail.

### `cogames play [MISSION] [POLICY]`

Play an episode of the specified mission. Cogs' actions are determined by the provided policy.

**Options:**

- `--steps N`: Number of steps (default: 1000)
- `--render MODE`: 'gui' or 'text' (default: gui)
- `--interactive`: Interactive mode (default: true)

`cogames play` supports a gui-based and text-based game renderer, both of which support many features to inspect agents
and manually play alongside them.

### `cogames train [MISSION] [POLICY]`

Train a policy on a mission.

**Options:**
- `--steps N`: Training steps (default: 10000)
- `--device STR`: 'auto', 'cpu', or 'cuda' (default: auto)
- `--batch-size N`: Batch size (default: 4096)
- `--num-workers N`: Worker processes (default: CPU count)

#### Training curricula

Passing one of the rotation curricula as the `mission` argument cycles training across the six training facilities and the `machina_1` and `machina_2` maps. These suppliers refresh the map each time `cogames train` requests a new environment so policies see a steady mix of layouts.

- `training_rotation`: Baseline rotation with the standard heart recipe and reward structure.
- `training_rotation_easy`: Enables the "easy" heart recipe in each map and extends the episode length, reducing the component requirements so current architectures can reliably craft hearts even when the default recipe is out of reach.
- `training_rotation_shaped`: Adds the shaped intermediate rewards and longer episodes while keeping the full recipe, providing denser feedback without altering the target objective.
- `training_rotation_easy_shaped`: Combines the easier heart recipe and shaped rewards, plus extended episode lengths, letting agents practice end-to-end heart crafting in a forgiving setting.

### Custom Policy Architectures

To get started, `cogames` supports some torch-nn-based policy architectures out of the box (such as SimplePolicy). To
supply your own, you will want to extend `cogames.policy.Policy`.

```python
from cogames.policy.interfaces import Policy

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

To train with using your class, supply a path to it in your POLICY argument, e.g.
`cogames train training_facility_1 path.to.MyPolicy`.

#### Environment API

The underlying environment follows the Gymnasium API:

```python
from cogames.game import get_mission
from mettagrid.envs import MettaGridEnv

# Load a mission configuration
config, _, __ = game_module.get_mission("assembler_2_complex", "default")

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

### `cogames eval [MISSION] [POLICIES...]`

Evaluate one or more policies. Note that here, you can provide a list of `POLICY` arguments if you want to run evaluations on mixed-policy populations.

**Examples:**

```bash
# Evaluate a single trained policy checkpoint
cogames eval machina_1 simple:train_dir/model.pt

# Mix two policies: 3 parts your policy, 5 parts random policy
cogames eval machina_1 simple:train_dir/model.pt:3 random::5
```

**Options:**

- `--episodes N`: Number of episodes (default: 10)
- `--action-timeout-ms N`: Timeout per action (default: 250ms)

When multiple policies are provided, `cogames eval` fixes the number of agents each policy will control, but
randomizes their assignments each episode.

### `cogames make-mission [BASE_MISSION]`

Create custom mission configuration. In this case, the mission provided is the template mission to which you'll apply modifications.

**Options:**

- `--agents N`: Number of agents (default: 2)
- `--width W`: Map width (default: 10)
- `--height H`: Map height (default: 10)
- `--output PATH`: Save to file

You will be able to provide your specified `--output` path as the `MISSION` argument to other `cogames` commmands.

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
