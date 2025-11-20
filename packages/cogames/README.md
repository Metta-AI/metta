# CoGames: A Game Environment for the Alignment League Benchmark

CoGames is the game environment for Softmax's
[Alignment League Benchmark (ALB)](https://www.softmax.com/alignmentleague) — a suite of multi-agent games designed to
measure how well AI agents align, coordinate, and collaborate with others (both AIs and humans).

The first ALB game, Cogs vs Clips, is implemented entirely within the CoGames environment. You can create your own
policy and submit it to our benchmark/pool.

## The game: Cogs vs Clips

Cogs vs Clips is a cooperative production-and-survival game where teams of AI agents ("Cogs") work together on the
asteroid Machina VII. Their mission: Produce and protect **HEARTs** (Holon Enabled Agent Replication Templates) by
gathering resources, operating machinery, and assembling components. Success is impossible alone! Completing these
missions requires multiple cogs working in tandem.

<p align="middle">
<img src="assets/showoff.gif" alt="Example Cogs vs Clips video">
<br>

There are many mission configurations available, with different map sizes, resource and station layouts, and game rules.
Cogs should refer to their [MISSION.md](MISSION.md) for a thorough description of the game mechanics. Overall, Cogs vs
Clips aims to present rich environments with:

- **Resource management**: Energy, materials (carbon, oxygen, germanium, silicon), and crafted components
- **Station-based interactions**: Different stations provide unique capabilities (extractors, assemblers, chargers,
  chests)
- **Sparse rewards**: Agents receive rewards only upon successfully crafting target items (hearts)
- **Partial observability**: Agents have limited visibility of the environment
- **Required multi-agent cooperation**: Agents must coordinate to efficiently use shared resources and stations, while
  only communicating through movement and emotes (❤️, 🔄, 💯, etc.)

Once your policy is successfully assembling hearts, submit it to our Alignment League Benchmark. ALB evaluates how your
policy plays with other policies in the pool through running multi-policy, multi-agent games. Our focal metric is VORP
(Value Over Replacement Policy), an estimate of how much your agent improves team performance in scoring hearts.

## Quick Start

Upon installation, try playing cogames with our default starter policies as Cogs. Use `cogames policies` to see a full
list of default policies.

```bash
# We recommend using a virtual env
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv .venv
source .venv/bin/activate

# Install cogames
uv pip install cogames

# List available missions
cogames missions

# Describe a specific mission in detail
cogames missions -m [MISSION]

# List available variants for modifying missions
cogames variants

# List all missions used as evals for analyzing the behaviour of agents
cogames evals

# Shows all policies available and their shorthands
cogames policies

# Authenticate before submitting or checking leaderboard
cogames login

# Inspect your leaderboard submissions
cogames submissions

# Show current leaderboard
cogames leaderboard

# Show version info for the installed tooling stack
cogames version
```

## Easy Mode - Best for getting started

Let's walk through playing an easy mission in Cogs vs. Clips, then training a simple starter policy. `easy_mode` uses
three variants to simplify training:

- `lonely_heart` - Simplifies heart crafting to require only 1 of each resource (carbon, oxygen, germanium, silicon,
  energy)
- `heart_chorus` - Provides reward shaping that gives bonuses for gaining hearts and maintaining diverse inventories
- `pack_rat` - Raises all capacity limits (heart, cargo, energy, gear) to 255 so agents never run out of storage space

```bash
# Play an episode yourself
cogames tutorial

# Play an episode of the easy_mode mission with a scripted policy
cogames play -m easy_mode -p baseline

# Try the scripted policy on a set of eval missions
cogames eval -set integrated_evals -p baseline

# Train with an LSTM policy on easy_mode
cogames train -m easy_mode -p lstm
```

## Play, Train, and Eval

Most commands are of the form `cogames <command> -m [MISSION] -p [POLICY] [OPTIONS]`

To specify a `MISSION`, you can:

- Use a mission name from the registry given by `cogames missions`, e.g. `training_facility_1`.
- Use a path to a mission configuration file, e.g. `path/to/mission.yaml`.
- Alternatively, specify a set of missions with `-set` or `-S`.

To specify a `POLICY`, provide an argument with up to three parts `CLASS[:DATA][:PROPORTION]`:

- `CLASS`: Use a policy shorthand or full path from the registry given by `cogames policies`, e.g. `lstm` or
  `cogames.policy.random.RandomPolicy`.
- `DATA`: Optional path to a weights file or directory. When omitted, defaults to the policy's built-in weights.
- `PROPORTION`: Optional positive float specifying the relative share of agents that use this policy (default: 1.0).

### `cogames play -m [MISSION] -p [POLICY]`

Play an episode of the specified mission.

Cogs' actions are determined by the provided policy, except if you take over their actions manually.

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

By default, our `stateless` policy architecture will be used. But as is explained above, you can select a different
policy architecture we support out of the box (like `lstm`), or can define your own and supply a path to it.

Any policy provided must implement the `TrainablePolicy` interface, which you can find in
`cogames/policy/interfaces.py`.

You can continue training an already-initialized policy by also supplying a path to its weights checkpoint file:

```
cogames train -m [MISSION] -p path/to/policy.py:train_dir/my_checkpoint.pt
```

Note that you can supply repeated `-m` missions. This yields a training curriculum that rotates through those
environments:

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
supply your own, extend the canonical `cogames.policy.MultiAgentPolicy` base class.

```python
from cogames.policy import MultiAgentPolicy

class MyPolicy(MultiAgentPolicy):
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

Evaluate one or more policies on one or more missions.

We provide a set of eval missions which you can use instead of missions `-m`. Specify `-set` or `-S` among:
`eval_missions`, `integrated_evals`, `spanning_evals`, `diagnostic_evals`, `all`.

You can provide multiple `-p POLICY` arguments if you want to run evaluations on mixed-policy populations.

**Examples:**

```bash
# Evaluate a single trained policy checkpoint
cogames eval -m machina_1 -p stateless:train_dir/model.pt

# Evaluate a single trained policy across a mission set with multiple agents
cogames eval -set integrated_evals -p stateless:train_dir/model.pt

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

Create a custom mission configuration. In this case, the mission provided is the template mission to which you'll apply
modifications.

**Options:**

- `--agents N`: Number of agents (default: 2)
- `--width W`: Map width (default: 10)
- `--height H`: Map height (default: 10)
- `--output PATH`: Save to file

You will be able to provide your specified `--output` path as the `MISSION` argument to other `cogames` commands.

## Policy Submission

### `cogames login`

Make sure you have authenticated before submitting a policy.

### `cogames submit -p [POLICY] -n [NAME]`

**Options:**

- `--include-files`: Can be specified multiple times, such as --include-files file1.py --include-files dir1/
- `--dry-run`: Validates the policy works for submission without uploading it

When a new policy is submitted, it is queued up for evals with other policies, both randomly selected and designated
policies for the Alignment League Benchmark.

## Citation

If you use CoGames in your research, please cite:

```bibtex
@software{cogames2025,
  title={CoGames: Multi-Agent Cooperative Game Environments},
  author={Metta AI},
  year={2025},
  url={https://github.com/metta-ai/metta}
}
```
