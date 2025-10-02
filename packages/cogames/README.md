# CoGames: Cogs vs Clips Multi-Agent RL Environment

CoGames is a collection of multi-agent cooperative and competitive environments designed for reinforcement learning
research. The primary focus is the **Cogs vs Clips** competition - a challenging multi-agent resource management and
assembly game built on the MettagGrid framework.

## 🎮 Cogs vs Clips Competition

In Cogs vs Clips, multiple "Cog" agents must cooperate to gather resources, operate machinery, and assemble components
to achieve objectives. The environment features:

- **Multi-agent cooperation**: Agents must coordinate to efficiently use shared resources and stations
- **Resource management**: Energy, materials (carbon, oxygen, germanium, silicon), and crafted components
- **Station-based interactions**: Different stations provide unique capabilities (extractors, assemblers, chargers,
  chests)
- **Sparse rewards**: Agents receive rewards only upon successfully crafting target items (hearts)
- **Partial observability**: Agents have limited visibility of the environment

### Game Mechanics

**Resources:**

- `energy`: Consumed for movement and operating extractors
- `carbon`, `oxygen`, `germanium`, `silicon`: Base materials extracted from stations
- `heart`: The target objective item
- `decoder`, `modulator`, `resonator`, `scrambler`: Advanced components

**Station Types:**

- **Charger**: Provides energy to agents
- **Extractors** (Carbon/Oxygen/Geranium/Silicon): Convert energy into materials
- **Assembler**: Combines resources to create components or objectives
- **Chest**: Storage for resource sharing between agents

## 🚀 Quick Start

### Installation

```bash
# Install the package
uv pip install cogames
```

### Running Your First Game

```bash
# List all available games
cogames games

# Play a simple single-agent assembler scenario
cogames play assembler_1_simple --steps 100 --render

# Play a multi-agent scenario
cogames play assembler_2_complex --steps 200 --render

# Run without rendering for faster execution
cogames play machina_2 --no-render --steps 500
```

## 🤖 For RL Researchers

### Training a Policy

CoGames integrates with standard RL training frameworks. Currently supports:

- PPO (Proximal Policy Optimization)
- A2C (Advantage Actor-Critic)
- DQN (Deep Q-Networks)

```bash
# Train a PPO agent on a single-agent scenario
cogames train machina_1 --steps 50000
```

### Evaluating Policies

```bash
# Evaluate a specific trained policy checkpoint
cogames evaluate machina_1 --policy simple --policy-data ./train_dir/my_policy.pt

# Evaluate the latest policy checkpoint in a directory
cogames evaluate machina_1 --policy simple --policy-data ./train_dir/

# Baseline comparison with random policy
cogames evaluate machina_1 --policy random


# Evaluate many policies
# Accepts a list of policy specs, each in the format
#   {policy class shorthand or path}:{optional proportion, default 1}:{optional policy data path}
cogames evaluate-many machina_1 simple:1:./train_dir/my_policy1.pt simple:2:./train_dir/my_policy2.pt random:5
```

### Implementing Custom Policies

Create your own policy by extending the `Policy` base class:

```python
from cogames.policy import Policy
from typing import Any, Optional
import torch
import torch.nn as nn

class MyCustomPolicy(Policy):
    def __init__(self, observation_space, action_space):
        self.network = MyNeuralNetwork(observation_space, action_space)

    def get_action(self, observation: Any, agent_id: Optional[int] = None) -> Any:
        """Compute action from observation."""
        with torch.no_grad():
            action_logits = self.network(observation)
            action = torch.argmax(action_logits).item()
        return action

    def reset(self) -> None:
        """Reset any internal state (e.g., RNN hidden states)."""
        pass

    def save(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save(self.network.state_dict(), path)

    @classmethod
    def load(cls, path: str, env=None) -> "MyCustomPolicy":
        """Load model from checkpoint."""
        policy = cls(env.observation_space, env.action_space)
        policy.network.load_state_dict(torch.load(path))
        return policy
```

### Environment API

The underlying MettagGrid environment follows the Gymnasium API:

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

## 📊 Available Scenarios

### Tutorial Scenarios

- `assembler_1_simple`: Single agent, simple assembly recipe
- `assembler_1_complex`: Single agent, complex recipes
- `assembler_2_simple`: 4 agents, simple cooperation
- `assembler_2_complex`: 4 agents, complex cooperation

### Competition Scenarios

- `machina_1`: Single agent, full game mechanics
- `machina_2`: 4 agents, full game mechanics

Use `cogames games [scenario_name]` for detailed information about each scenario.

## 🔧 Creating Custom Scenarios

```python
from cogames.cogs_vs_clips.scenarios import make_game

# Create a custom game configuration
config = make_game(
    num_cogs=4,                    # Number of agents
    num_assemblers=2,              # Number of assembler stations
    num_chargers=1,                # Energy stations
    num_carbon_extractors=1,       # Material extractors
    num_oxygen_extractors=1,
    num_germanium_extractors=1,
    num_silicon_extractors=1,
    num_chests=2,                  # Storage chests
)

# Modify map size
config.game.map_builder.width = 15
config.game.map_builder.height = 15

# Save configuration
cogames make-scenario --name my_scenario --agents 4 --width 15 --height 15 --output my_scenario.yaml
```

## 🏆 Competition Tips

1. **Coordination is Key**: Multi-agent scenarios require effective coordination. Consider:
   - Task allocation strategies
   - Communication through glyph changes
   - Resource sharing via chests

2. **Energy Management**: Energy is limited and required for most actions:
   - Plan efficient paths
   - Use chargers strategically
   - Balance exploration vs exploitation

3. **Hierarchical Planning**: Break down the assembly task:
   - Gathering phase (collect base materials)
   - Processing phase (operate extractors)
   - Assembly phase (combine at assemblers)

4. **Curriculum Learning**: Start with simpler scenarios:
   - Master single-agent tasks first
   - Graduate to multi-agent coordination
   - Increase complexity gradually

## 🔬 Research Integration

CoGames is designed to integrate with the Metta RL framework:

```bash
# Using Metta's recipe system for advanced training
uv run ./tools/run.py experiments.recipes.cogames.train scenario=assembler_2_complex

# Distributed training with Metta
uv run ./tools/run.py experiments.recipes.cogames.distributed_train \
    num_workers=4 \
    scenario=machina_2
```

## 📚 Additional Resources

- **MettagGrid Documentation**: The underlying grid world engine
- **Metta RL Framework**: Advanced training recipes and algorithms
- **Competition Leaderboard**: Track your progress against other researchers

## 🐛 Debugging and Visualization

```bash
# Interactive mode for debugging
cogames play machina_1 --interactive

# Use the text renderer
cogames play machina_1 --interactive --render text

# Non-interactive step-by-step execution
cogames play machina_2 --steps 10
```

## 📝 Citation

If you use CoGames in your research, please cite:

```bibtex
@software{cogames2024,
  title={CoGames: Multi-Agent Cooperative Game Environments},
  author={Metta AI},
  year={2024},
  url={https://github.com/metta-ai/metta}
}
```

## 💡 Support

For questions about the Cogs vs Clips competition or CoGames environments:

- Open an issue in the repository
- Contact the competition organizers
- Check the competition Discord channel
