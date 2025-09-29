# CoGames: Cogs vs Clips Multi-Agent RL Environment

CoGames is a collection of multi-agent cooperative and competitive environments designed for reinforcement learning
research. The primary focus is the **Cogs vs Clips** competition - a challenging multi-agent resource management and
assembly game built on the MettagGrid framework.

## 🎮 Cogs vs Clips Competition

In Cogs vs Clips, multiple "Cog" agents must cooperate to gather resources, operate machinery, and assemble components
to achieve objectives. The environment features:

- **Multi-agent cooperation**: Agents must coordinate to efficiently use shared resources and stations
- **Resource management**: Energy, materials (carbon, oxygen, geranium, silicon), and crafted components
- **Station-based interactions**: Different stations provide unique capabilities (extractors, assemblers, chargers,
  chests)
- **Sparse rewards**: Agents receive rewards only upon successfully crafting target items (hearts)
- **Partial observability**: Agents have limited visibility of the environment

### Game Mechanics

**Resources:**

- `energy`: Consumed for movement and operating extractors
- `carbon`, `oxygen`, `geranium`, `silicon`: Base materials extracted from stations
- `heart`: The target objective item
- `disruptor`, `modulator`, `resonator`, `scrabbler`: Advanced components

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

`cogames train` launches PuffeRL with a CoGames environment. Important flags:

- `--policy` selects the policy class (defaults to `SimplePolicy`).
- `--use-rnn` enables recurrent policies such as `StatefulPolicy`.
- `--curriculum module.symbol` loads a Python iterable or generator that yields `MettaGridConfig` instances for curricula.
- `--vector-backend {multiprocessing,serial,ray}` chooses the vector environment implementation. Use `serial` for lightweight local runs or `ray` for distributed sampling when Ray is installed.
- `--num-envs`, `--num-workers`, `--batch-size`, and `--minibatch-size` tune rollout throughput. When omitted, the batch size defaults to `num_envs * 32`.
- `--initial-weights` accepts either a specific checkpoint file or a directory; directories automatically load the newest `.pt/.pth/.ckpt` file.
- `--checkpoint-interval` controls how frequently PuffeRL writes checkpoints into `--checkpoints`.

Examples:

```bash
# Minimal CPU PPO run that saves checkpoints into ./runs/basic
cogames train assembler_1_simple \
  --device cpu \
  --steps 2000 \
  --num-envs 2 \
  --num-workers 1 \
  --batch-size 128 \
  --minibatch-size 128 \
  --checkpoints ./runs/basic

# Stateful policy with a curriculum and the serial backend
cogames train --curriculum myproject.curricula.cogs_vs_clips \
  --policy cogames.examples.stateful_policy.StatefulPolicy \
  --use-rnn \
  --vector-backend serial \
  --steps 5000 \
  --checkpoints ./runs/curriculum

# Multi-GPU run with torchrun (rank-aware seeding and device assignment)
torchrun --standalone --nproc-per-node=2 -m cogames.main train \
  assembler_1_simple --device cuda --num-envs 32 --num-workers 8
```

Every CLI command also accepts a global `--timeout` flag. Set it to automatically abort long-running invocations (useful in CI or quick smoke tests).

### Policy Bundles

Use `cogames policy` utilities to package checkpoints with their class paths and reload them later:

```bash
# Bundle an existing checkpoint into ./bundles/simple
cogames policy export \
  cogames.examples.simple_policy.SimplePolicy \
  ./runs/basic/policy.pt \
  ./bundles/simple

# Instantiate the bundled policy for a scenario
cogames policy load ./bundles/simple assembler_1_simple --device cpu
```

### Evaluating Policies

> **Note:** Evaluation support is still under construction. The `cogames evaluate` command currently prints a placeholder message.

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

### Observation Space

Observations are dictionaries containing:

- `rgb`: RGB image of the agent's view (H×W×3)
- `inventory`: Agent's current resources
- `position`: Agent's (x, y) coordinates
- `orientation`: Agent's facing direction
- `glyph`: Agent's current communication symbol

### Action Space

Available actions:

- `0`: No-op (do nothing)
- `1-4`: Move (forward, backward, left, right)
- `5-8`: Rotate (turn left/right)
- `9-24`: Change glyph (for communication)
- Additional actions may be available based on scenario

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
    num_geranium_extractors=1,
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
cogames play assembler_1_simple --interactive --render

# Step-by-step execution
cogames play machina_2 --steps 10 --render --interactive
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
