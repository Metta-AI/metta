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

# List available missions
cogames missions

# Play an episode of the training_facility_1 mission
cogames play -m training_facility_1 -p random

# Train a policy in that environment using an out-of-the-box, stateless network architecture
cogames train -m training_facility_1 -p stateless

# Watch or play along side your trained policy
cogames play -m training_facility_1 -p stateless:train_dir/policy.pt

# Evaluate how your policy performs on a different mission
cogames eval -m machina_1 -p stateless:./train_dir/policy.pt
```

## Evaluation Reference

### Scripted Agent Hyperparameter Presets

Scripted evaluation policies use the streamlined `Hyperparameters` dataclass from `cogames.policy.scripted_agent.hyperparameters`. Presets are defined in `hyperparameter_presets.py` and exposed via `HYPERPARAMETER_PRESETS`.

- `story_mode` (`sequential_baseline` alias)
  - Strategy tuned for showcase runs; disables probes, extends exploration to 120 steps, and raises recharge thresholds (`start_small=60`, `start_large=45`).
  - Keeps patience high (`wait_if_cooldown_leq=5`, `max_patience_steps=20`) and enforces expansive resource caps (`resource_focus_limits={'carbon':4,'oxygen':4,'germanium':2,'silicon':2}`).
  - Maintains conservative energy policy: always charge to full and requires only 60 energy before routing for silicon.
- `courier` (`balanced` alias)
  - Fast routing baseline with shorter exploration (60 steps) and tighter patience controls (`wait_if_cooldown_leq=3`, `max_patience_steps=10`).
  - Prefers partial recharges (`recharge_until_full=False`) and drops small recharge thresholds to 40/30.
  - Limits simultaneous extractor focus to 3/3/2/2 resources.
- `scout` (`explorer_long` alias)
  - Probe-heavy preset (`use_probes=True`) with expanded probe radius (18 tiles), 12 concurrent targets, and long revisit cooldowns.
  - Longest exploration phase (160 steps) paired with aggressive movement (`wait_if_cooldown_leq=1`) and lower recharge targets (30/20 start, 80/85 stop).
  - Keeps focus balanced at 2 extractors per resource type.
- `hoarder` (`greedy_conservative`, `efficiency_heavy` aliases)
  - High patience, stockpiling behaviour (`wait_if_cooldown_leq=8`, `max_patience_steps=30`, `patience_multiplier=2.0`).
  - Demands plenty of energy before risky actions (`min_energy_for_silicon=80`) and always recharges to full.
  - Allows the widest resource focus limits (5/5/3/3) and maintains depletion threshold at 0.35.

> All presets inherit defaults such as `strategy_type`, probe configuration, patience tuning, and energy management parameters. Legacy factory helpers (e.g. `create_aggressive_preset`) resolve to these same presets for backward compatibility.

### Difficulty Variants

Difficulty settings live in `cogames.cogs_vs_clips.evals.difficulty_variants`. Each `DifficultyLevel` tweaks extractor max uses, efficiency, passive energy, and optionally mission-wide limits. Aliases (`easy`, `medium`, `extreme`) map to `story_mode`, `standard`, and `brutal` respectively.

- `story_mode`
  - 12 guaranteed uses per extractor, generous efficiency (140), charger efficiency 150, passive energy regen 2.
  - Disables agent-count scaling to keep showcase behaviour deterministic.
- `standard`
  - Baseline figures inherited from the mission; serves as the default evaluation (`medium` alias).
- `hard`
  - Caps extractors at 4/4/6/3 uses, trims efficiency (80/65/75/70), zeroes passive regen, and increases move cost to 3.
  - Disables agent-count scaling helpers to preserve challenge.
- `brutal` (`extreme` alias)
  - Severe scarcity: 2/2/3/2 uses, low efficiency (55/45/50/50), charger efficiency 60, no regen.
  - Reduces agent inventory caps (energy 70, cargo 80) and raises move cost to 3.
- `single_use`
  - Every extractor is single-shot; chargers remain strong (efficiency 120) with minimal passive regen (1).
- `speed_run`
  - Emphasises tempo: 6 uses per extractor, high efficiency (160 across resources and chargers), move cost 1, and shortens missions to 600 steps.
  - Keeps agent-scaling enabled so larger teams get proportional resources.
- `energy_crisis`
  - No passive regen and weak chargers (efficiency 50); mission otherwise inherits baseline limits.

All scaling-enabled difficulties ensure extractor counts and efficiency rise with agent count, clamp charger efficiency to at least 50, and keep any non-zero passive regen at or above 1 to retain solvability.

## Commands

Most commands are of the form `cogames <command> -p [MISSION] -p [POLICY] [OPTIONS]`

To specify a `MISSION`, you can:

- Use a mission name from the default registry emitted by `cogames missions`, e.g. `training_facility_1`
- Use a path to a mission configuration file, e.g. path/to/mission.yaml"

To specify a `POLICY`, provide an argument with up to three parts `CLASS[:DATA][:PROPORTION]`:

- `CLASS`: Policy shorthand (`noop`, `random`, `lstm`, `stateless`) or fully qualified class path like
  `cogames.policy.random.RandomPolicy`.
- `DATA`: Optional path to a weights file or directory. When omitted, defaults to the policy's built-in weights.
- `PROPORTION`: Optional positive float specifying the relative share of agents that use this policy (default: 1.0).

### `cogames missions -m [MISSION]`

Lists all missions and their high-level specs.

If a mission is provided, it describe a specific mission in detail.

### `cogames play -m [MISSION] -p [POLICY]`

Play an episode of the specified mission.

**Policy** Cogs' actions are determined by the provided policy, except if you take over their actions manually.

If not specified, this command will use the `noop`-policy agent -- do not be surprised if when you play you don't see
other agents moving around! Just provide a different policy, like `random`.

**Options:**

- `--steps N`: Number of steps (default: 1000)
- `--render MODE`: 'gui' or 'text' (default: gui)
- `--non-interactive`: Non-interactive mode (default: false)

`cogames play` supports a gui-based and text-based game renderer, both of which support many features to inspect agents
and manually play alongside them.

### `cogames train -m [MISSION] -p [POLICY]`

Train a policy on a mission.

**Policy** By default, our `stateless` policy architecture will be used. But as is explained above, you can select a
different policy architecture we support out of the box (like `lstm`), or can define your own and supply a path to it.

Any policy provided must implement the `TrainablePolicy` interface, which you can find in
`cogames/policy/interfaces.py`.

You can continue training an already-initialized policy by also supplying a path to its weights checkpoint file:

```
cogames train -m [MISSION] -p path/to/policy.py:train_dir/my_checkpoint.pt
```

**Mission** Note that you can supply repeated `-m` missions. This yields a training curriculum that rotates through
those environments:

```
cogames train -m training_facility_1 -m training_facility_2 -p stateless
```

You can also specify multiple missions with `*` wildcards:

- `cogames train -m 'machina_2_bigger:*'` will specify all missions on the machina_2_bigger map
- `cogames train -m '*:shaped'` will specify all "shaped" missions across all maps
- `cogames train -m 'machina*:shaped'` will specify all "shaped" missions on all machina maps

**Options:**

- `--steps N`: Training steps (default: 10000)
- `--device STR`: 'auto', 'cpu', or 'cuda' (default: auto)
- `--batch-size N`: Batch size (default: 4096)
- `--num-workers N`: Worker processes (default: CPU count)

### Custom Policy Architectures

To get started, `cogames` supports some torch-nn-based policy architectures out of the box (such as StatelessPolicy). To
supply your own, you will want to extend `cogames.policy.Policy`.

```python
from mettagrid.policy.policy import MultiAgentPolicy as Policy

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
from cogames.cli.mission import get_mission
from mettagrid import PufferMettaGridEnv
from mettagrid.simulator import Simulator

# Load a mission configuration
_, config = get_mission("assembler_2_complex")

# Create environment
simulator = Simulator()
env = PufferMettaGridEnv(simulator, config)

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

### `cogames eval -m [MISSION] [-m MISSION...] -p POLICY [-p POLICY...]`

Evaluate one or more policies on one more more missions

**Policy** Note that here, you can provide multiple `-p POLICY` arguments if you want to run evaluations on mixed-policy
populations.

**Examples:**

```bash
# Evaluate a single trained policy checkpoint
cogames eval -m machina_1 -p stateless:train_dir/model.pt

# Mix two policies: 3 parts your policy, 5 parts random policy
cogames eval -m machina_1 -p stateless:train_dir/model.pt:3 -p random::5
```

**Options:**

- `--episodes N`: Number of episodes per mission (default: 10)
- `--action-timeout-ms N`: Timeout per action (default: 250ms)
- `--steps N`: Max steps per episode
- `--format [json/yaml]`: Output results as structured json or yaml (default: None for human-readable tables)

When multiple policies are provided, `cogames eval` fixes the number of agents each policy will control, but randomizes
their assignments each episode.

### `cogames make-mission -m [BASE_MISSION]`

Create custom mission configuration. In this case, the mission provided is the template mission to which you'll apply
modifications.

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
