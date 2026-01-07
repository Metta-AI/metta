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
<img src="assets/cvc-reel.gif" alt="Example Cogs vs Clips">
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

# Get Started

## Step 1: Set up and install

Install [cogames](https://pypi.org/project/cogames/) as a Python package.

```bash
# We recommend using a virtual env
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv .venv
source .venv/bin/activate

# Install cogames
uv pip install cogames
```

## Step 2: Game tutorial

Play an easy mission in Cogs vs. Clips using `cogames tutorial play`. Follow the instructions given in the terminal,
while you use the GUI to accomplish your first training mission.

## Step 3: Train a simple policy

We'll train a simple starter policy on `training_facility.harvest`. Optional variants to simplify training:

- `lonely_heart` - Simplifies heart crafting to require only 1 of each resource (carbon, oxygen, germanium, silicon,
  energy)
- `heart_chorus` - Provides reward shaping that gives bonuses for gaining hearts and maintaining diverse inventories
- `pack_rat` - Raises all capacity limits (heart, cargo, energy, gear) to 255 so agents never run out of storage space

```bash
# Play an episode of the training_facility.harvest mission with a scripted policy
cogames play -m training_facility.harvest -p class=baseline

# Try the scripted policy on a set of eval missions
cogames run -S integrated_evals -p class=baseline

# Train with an LSTM policy on training_facility.harvest
cogames tutorial train -m training_facility.harvest -p class=lstm
```

## Step 4: Learn about missions

Get familiar with different missions in Cogs vs. Clips so you can develop a policy that's able to handle different
scenarios.

Useful commands to explore:

```bash
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

# Authenticate before uploading or checking leaderboard
cogames login

# List available tournament seasons
cogames seasons

# View leaderboard for a season
cogames leaderboard --season SEASON

# View your tournament submissions
cogames submissions
```

# Develop a Policy

A **policy** contains the decision-making logic that controls your agents. Given an observation of the game state, a
policy outputs an action.

CoGames asks that policies implement the `MultiAgentPolicy` interface. Any implementation will work, and we provide two
templates to get you up and running:

- `cogames tutorial make-policy --scripted` gives a starter template for a simple, rule-based script
- `cogames tutorial make-policy --trainable` gives a basic neural-net based implementation that can be trained via
  `cogames tutorial train`

## Play, Train, and Run

Most commands are of the form `cogames <command> -m [MISSION] -p [POLICY] [OPTIONS]`

To specify a `MISSION`, you can:

- Use a mission name from the registry given by `cogames missions`, e.g. `training_facility_1`.
- Use a path to a mission configuration file, e.g. `path/to/mission.yaml`.
- Alternatively, specify a set of missions with `-S` or `--mission-set`.

To specify a `POLICY`, use one of two formats:

**URI format** (for checkpoint bundles):

- Point directly at a checkpoint bundle (directory or `.zip` containing `policy_spec.json`)
- Examples: `./train_dir/my_run:v5`, `./train_dir/my_run:v5.zip`, `s3://bucket/path/run:v5.zip`
- Use `:latest` suffix to auto-resolve the highest epoch: `./train_dir/checkpoints:latest`

**Key-value format** (for explicit class + weights):

- `class=`: Policy shorthand or full class path from `cogames policies`, e.g. `class=lstm` or
  `class=cogames.policy.random.RandomPolicy`.
- `data=`: Optional path to a weights file (e.g., `weights.safetensors`). Must be a file, not a directory.
- `proportion=`: Optional positive float specifying the relative share of agents that use this policy (default: 1.0).
- `kw.<arg>=`: Optional policy `__init__` keyword arguments (all values parsed as strings).

### `cogames play -m [MISSION] -p [POLICY]`

Play an episode of the specified mission.

Cogs' actions are determined by the provided policy, except if you take over their actions manually.

If not specified, this command will use the `noop`-policy agent -- do not be surprised if when you play you don't see
other agents moving around! Just provide a different policy, like `random`.

**Options:**

- `--steps N`: Number of steps (default: 10000)
- `--render MODE`: 'gui' or 'text' (default: gui)
- `--non-interactive`: Non-interactive mode (default: false)

`cogames play` supports a gui-based and text-based game renderer, both of which support many features to inspect agents
and manually play alongside them.

### `cogames tutorial train -m [MISSION] -p [POLICY]`

Train a policy on a mission.

By default, our `stateless` policy architecture will be used. But as is explained above, you can select a different
policy architecture we support out of the box (like `lstm`), or can define your own and supply a path to it.

Any policy provided must implement the `MultiAgentPolicy` interface with a trainable `network()` method, which you can
find in `mettagrid/policy/policy.py`.

You can continue training from a checkpoint bundle (use URI format):

```
cogames tutorial train -m [MISSION] -p ./train_dir/my_run:v5
```

Or load weights into an explicit class:

```
cogames tutorial train -m [MISSION] -p class=path.to.MyPolicy,data=train_dir/run:v5/weights.safetensors
```

Note that you can supply repeated `-m` missions. This yields a training curriculum that rotates through those
environments:

```
cogames tutorial train -m training_facility_1 -m training_facility_2 -p class=stateless
```

You can also specify multiple missions with `*` wildcards:

- `cogames tutorial train -m 'machina_2_bigger:*'` will specify all missions on the machina_2_bigger map
- `cogames tutorial train -m '*:shaped'` will specify all "shaped" missions across all maps
- `cogames tutorial train -m 'machina*:shaped'` will specify all "shaped" missions on all machina maps

**Options:**

- `--steps N`: Training steps (default: 10000)
- `--device STR`: 'auto', 'cpu', or 'cuda' (default: auto)
- `--batch-size N`: Batch size (default: 4096)
- `--num-workers N`: Worker processes (default: CPU count)

### Custom Policy Architectures

CoGames supports torch-based policy architectures out of the box (such as `stateless` and `lstm`). To create your own
trainable policy, run:

```bash
cogames tutorial make-policy --trainable -o my_policy.py
```

This generates a complete working template. See the
[trainable policy template](src/cogames/policy/trainable_policy_template.py) for the full implementation. The key
components are:

- **`MultiAgentPolicy`**: The main policy class that the training system interacts with
- **`AgentPolicy`**: Per-agent decision-making (returned by `agent_policy()`)
- **`network()`**: Returns the `nn.Module` for training (must implement `forward_eval(obs, state) -> (logits, values)`)
- **`load_policy_data()` / `save_policy_data()`**: Checkpoint serialization

To train using your policy:

```bash
cogames tutorial train -m training_facility.harvest -p class=my_policy.MyTrainablePolicy
```

#### Environment API

The underlying environment follows the Gymnasium API:

```python
from cogames.cli.mission import get_mission
from mettagrid import PufferMettaGridEnv
from mettagrid.simulator import Simulator

# Load a mission configuration
_, config = get_mission("machina_1.open_world")

# Create environment
simulator = Simulator()
env = PufferMettaGridEnv(simulator, config)

# Reset environment
obs, info = env.reset()

# Game loop
for step in range(10000):
    # Your policy computes actions for all agents
    actions = policy.get_actions(obs)  # Dict[agent_id, action]

    # Step environment
    obs, rewards, terminated, truncated, info = env.step(actions)

    if terminated or truncated:
        obs, info = env.reset()
```

### `cogames run -m [MISSION] [-m MISSION...] -p POLICY [-p POLICY...]`

Evaluate one or more policies on one or more missions.

We provide a set of eval missions which you can use instead of missions `-m`. Specify `-S` or `--mission-set` among:
`eval_missions`, `integrated_evals`, `spanning_evals`, `diagnostic_evals`, `all`.

You can provide multiple `-p POLICY` arguments if you want to run evaluations on mixed-policy populations.

**Examples:**

```bash
# Evaluate a checkpoint bundle
cogames run -m machina_1 -p ./train_dir/my_run:v5

# Evaluate across a mission set
cogames run -S integrated_evals -p ./train_dir/my_run:v5

# Mix two policies: 3 parts your policy, 5 parts random
cogames run -m machina_1 -p ./train_dir/my_run:v5,proportion=3 -p class=random,proportion=5
```

**Options:**

- `--episodes N`: Number of episodes per mission (default: 10)
- `--action-timeout-ms N`: Timeout per action (default: 250ms)
- `--steps N`: Max steps per episode
- `--format [json/yaml]`: Output results as structured json or yaml (default: None for human-readable tables)

When multiple policies are provided, `cogames run` fixes the number of agents each policy will control, but randomizes
their assignments each episode.

### `cogames make-mission -m [BASE_MISSION]`

Create a custom mission configuration. In this case, the mission provided is the template mission to which you'll apply
modifications.

**Options:**

- `--cogs N` or `-c N`: Number of cogs
- `--width W` or `-w W`: Map width
- `--height H` or `-h H`: Map height
- `--output PATH`: Save to file

Note: If `--cogs`, `--width`, or `--height` are not specified, the values come from the base mission's configuration.

You will be able to provide your specified `--output` path as the `MISSION` argument to other `cogames` commands.

## Policy Submission

### `cogames login`

Authenticate before uploading policies or viewing leaderboards.

### `cogames upload -p [POLICY] -n [NAME]`

Upload a policy to the server.

```
cogames submit -p ./train_dir/my_run:v5 -n my_policy
```

**Options:**

- `--include-files`: Additional files to include (can be specified multiple times)
- `--dry-run`: Validate the policy without uploading

### `cogames seasons`

List available tournament seasons you can submit to.

### `cogames submit [POLICY] --season [SEASON]`

Submit an uploaded policy to a tournament season.

```bash
# Submit latest version of a policy
cogames submit my_policy --season alb-season-1

# Submit a specific version
cogames submit my_policy:v3 --season alb-season-1
```

### `cogames submissions`

View your tournament submissions.

```bash
# View all submissions
cogames submissions

# Filter by season
cogames submissions --season alb-season-1

# Filter by policy (positional argument)
cogames submissions my_policy
```

### `cogames leaderboard --season [SEASON]`

View the leaderboard for a tournament season.

```bash
cogames leaderboard --season alb-season-1
```

When a policy is submitted to a season, it is queued for matches against other policies in that season's pools.

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
