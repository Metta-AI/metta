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
cogames play machina_3_big --no-render --steps 500
```

## 🤖 For RL Researchers

### Training a Policy

`cogames train` launches PuffeRL with a CoGames environment. Running `uv run cogames train` with no arguments now
cycles through the eight Machina biome maps (`machina_1` … `machina_7_big`) and exports them into the run directory so
subsequent commands reuse the same curriculum.

Useful flags (kept intentionally short):

- `--policy CLASS` selects the policy implementation (defaults to `SimplePolicy`). Add `--use-rnn` for recurrent
  policies.
- `--device VALUE` chooses the training device; `auto` prefers CUDA, then MPS, then CPU.
- `--num-envs`, `--num-workers`, `--batch-size`, `--minibatch-size` tune throughput. Defaults scale with detected
  hardware.
- `--curriculum module.symbol` loads a Python iterable/generator of `MettaGridConfig` objects instead of the default
  Machina set.
- `--run-dir PATH` changes where checkpoints live (`checkpoints/`) and where exported maps are written (`curricula/`).
- `--initial-weights PATH` and `--checkpoint-interval N` control PPO checkpointing.

Examples:

```bash
# Default biome sweep with all defaults
uv run cogames train

# Two-worker CPU run that keeps checkpoints in ./runs/assembler_experiment
uv run cogames train assembler_2_complex \
  --device cpu \
  --num-envs 4 \
  --num-workers 2 \
  --steps 2000 \
  --run-dir ./runs/assembler_experiment

# Curriculum-driven training with an LSTM policy
uv run cogames train \
  --curriculum myproject.curricula.cogs_vs_clips \
  --policy cogames.policy.lstm.LSTMPolicy \
  --use-rnn \
  --vector-backend serial \
  --steps 5000
```

Every CLI command also accepts a global `--timeout` flag. Set it to automatically abort long-running invocations (useful in CI or quick smoke tests).

### Exporting Curriculum Maps

Use `cogames curricula` to materialize game and curriculum configurations into a directory that can be consumed later by `cogames train` or other tools:

```bash
# Dump every built-in Cogs vs Clips scenario into packages/cogames/runs/default/curricula
uv run cogames curricula

# Train with default settings (exports Machina biome maps into the default run directory)
uv run cogames train

# Choose the destination explicitly and mix in a Python curriculum generator
uv run cogames curricula --output-dir ./runs/curricula/all_maps \
  --curriculum myproject.curricula.cogs_vs_clips

# Export a subset of games only
uv run cogames curricula --output-dir ./tmp/maps --game assembler_1_simple --game machina_3_big
```

The command writes each map configuration once (deduplicated by name) and prints the final output directory. `cogames train` automatically looks for maps in `packages/cogames/runs/default/curricula` when no `--curriculum` argument is provided.

### Generating Map Variants

Need a sweep of related maps? `cogames make-game` can generate a family of configurations by interpolating a configuration field across a range. The command accepts the same base options as before, plus:

- `--num-variants` – how many variants to generate (defaults to 1)
- `--key` – dotted path into the `MettaGridConfig` to modify (for example `game.map_builder.width`)
- `--min` / `--max` – inclusive range values for the sweep

When more than one variant is requested the `--output` argument must point to a directory; each variant is written as its own YAML file.

```bash
cogames make-game assembler_2_simple \
  --output ./runs/map_variants \
  --num-variants 3 \
  --key game.map_builder.width \
  --min 64 \
  --max 192
```

The example above creates three maps with widths evenly spaced between 64 and 192 cells and writes them into `./runs/map_variants/`.

### Cleaning Run Artifacts

If you need to reset the fallback run directories (for example, to clear out stale curricula that no longer match the latest game definitions), use `cogames clean`:

```bash
# Remove curricula and checkpoints from the default runs directory
uv run cogames clean

# Target a specific run directory and leave checkpoints untouched
uv run cogames clean --run-dir ./runs/assembler_1_simple --no-checkpoints

# Preview what would be deleted without modifying the filesystem
uv run cogames clean --dry-run
```

By default the command deletes both the `curricula/` and `checkpoints/` subdirectories. Use the boolean flags to fine-tune what gets removed.

### Policy Bundles

Use `cogames policy` utilities to package checkpoints with their class paths and reload them later:

```bash
# Bundle an existing checkpoint into ./bundles/simple
cogames policy export \
  cogames.policy.simple.SimplePolicy \
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

- `machina_1`: Compact biome map with full mechanics
- `machina_1_big`: Larger variant of `machina_1`
- `machina_2_bigger`: Expanded station density
- `machina_3_big`: Alternate layout with 500 stations
- `machina_4_bigger`: Alternate layout with 1000 stations
- `machina_5_big`: Third layout with 500 stations
- `machina_6_bigger`: Third layout with 1000 stations
- `machina_7_big`: Fourth layout with 500 stations

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
    scenario=machina_3_big
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
cogames play machina_4_bigger --steps 10 --render --interactive
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
