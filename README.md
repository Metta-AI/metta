# Metta AI

<p align="center">
  <a href="https://codecov.io/gh/Metta-AI/metta">
    <img src="https://codecov.io/gh/Metta-AI/metta/graph/badge.svg?token=SX28I8PS3E" alt="codecov">
  </a>
  <a href="https://github.com/Metta-AI/metta/actions/workflows/checks.yml">
    <img src="https://github.com/Metta-AI/metta/actions/workflows/checks.yml/badge.svg" alt="Tests">
  </a>
  <a href="https://discord.gg/secret-hologenesis">
    <img src="https://img.shields.io/discord/1309708848730345493?logo=discord&logoColor=white&label=Discord" alt="Discord">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License">
  </a>
  <a href="https://deepwiki.com/Metta-AI/metta">
    <img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki">
  </a>
</p>

A reinforcement learning codebase focusing on the emergence of cooperation and alignment in multi-agent AI systems.

- **Discord**: <https://discord.gg/mQzrgwqmwy>
- **Short (5m) Talk**: <https://www.youtube.com/watch?v=bt6hV73VA8I>
- **Talk**: <https://foresight.org/summary/david-bloomin-metta-learning-love-is-all-you-need/>

## What is Metta Learning?

<p align="middle">
<img src="docs/readme_showoff.gif" alt="Metta learning example video">
<br>
<a href="https://metta-ai.github.io/metta/?replayUrl=https%3A%2F%2Fsoftmax-public.s3.us-east-1.amazonaws.com%2Freplays%2Fandre_pufferbox_33%2Freplay.77200.json.z&play=true">Interactive demo</a>
</p>

Metta AI is an open-source research project investigating the emergence of cooperation and alignment in multi-agent AI
systems. By creating a model organism for complex multi-agent gridworld environments, the project aims to study the
impact of social dynamics, such as kinship and mate selection, on learning and cooperative behaviors of AI agents.

Metta AI explores the hypothesis that social dynamics, akin to love in biological systems, play a crucial role in the
development of cooperative AGI and AI alignment. The project introduces a novel reward-sharing mechanism mimicking
familial bonds and mate selection, allowing researchers to observe the evolution of complex social behaviors and
cooperation among AI agents. By investigating this concept in a controlled multi-agent setting, the project seeks to
contribute to the broader discussion on the path towards safe and beneficial AGI.

## Introduction

Metta is a simulation environment (game) designed to train AI agents capable of meta-learning general intelligence. The
core idea is to create an environment where incremental intelligence is rewarded, fostering the development of generally
intelligent agents.

### Motivation and Approach

1. **Agents and Environment**: Agents are shaped by their environment, learning policies that enhance their fitness. To
   develop general intelligence, agents need an environment where increasing intelligence is continually rewarded.

2. **Competitive and Cooperative Dynamics**: A game with multiple agents and some competition creates an evolving
   environment where challenges increase with agent intelligence. Purely competitive games often reach a Nash
   equilibrium, where locally optimal strategies are hard to deviate from. Adding cooperative dynamics introduces more
   behavioral possibilities and smooths the behavioral space.

3. **Kinship Structures**: The game features a flexible kinship structure, simulating a range of relationships from
   close kin to strangers. Agents must learn to coordinate with close kin, negotiate with more distant kin, and compete
   with strangers. This diverse social environment encourages continuous learning and intelligence growth.

The game is designed to evolve with the agents, providing unlimited learning opportunities despite simple rules.

### Game Overview

The current version of the game can be found [here](https://huggingface.co/metta-ai/baseline.v0.1.0). It's a grid world
with the following dynamics:

- **Agents and Vision**: Agents can see a limited number of squares around them.
- **Resources**: Agents harvest diamonds, convert them to energy at charger stations, and use energy to power the "heart
  altar" for rewards.
- **Energy Management**: All actions cost energy, so agents learn to manage their energy budgets efficiently.
- **Combat**: Agents can attack others, temporarily freezing the target and stealing resources.
- **Defense**: Agents can toggle shields, which drain energy but absorb attacks.
- **Cooperation**: Agents can share energy or resources and use markers to communicate.

### Exploration and Expansion

The game offers numerous possibilities for exploration, including:

1. **Diverse Energy Profiles**: Assigning different energy profiles to agents, essentially giving them different bodies
   and policies.
2. **Dynamic Energy Profiles**: Allowing agents to change their energy profiles, reflecting different postures or
   emotions.
3. **Resource Types and Conversions**: Introducing different resource types and conversion mechanisms.
4. **Environment Modification**: Enabling agents to modify the game board by creating, destroying, or altering objects.

### Kinship and Social Dynamics

The game explores various kinship structures:

1. **Random Kinship Scores**: Each pair of agents has a kinship score sampled from a distribution.
2. **Teams**: Agents belong to teams with symmetric kinship among team members.
3. **Hives/Clans/Families**: Structuring agents into larger kinship groups.

Future plans include incorporating mate-selection dynamics, where agents share future rewards at a cost, potentially
leading to intelligence gains through a signaling arms race.

Metta aims to create a rich, evolving environment where AI agents can develop general intelligence through continuous
learning and adaptation.

## Research Explorations

The project's modular design and open-source nature make it easy for researchers to adapt and extend the platform to
investigate their own hypotheses in this domain. The highly performant, open-ended game rules provide a rich environment
for studying these behaviors and their potential implications for AI alignment.

Some areas of research interest:

#### 1. Environment Development

Develop rich and diverse gridworld environments with complex dynamics, such as resource systems, agent diversity,
procedural terrain generation, support for various environment types, population dynamics, and kinship schemes.

#### 2. Agent Architecture Research

Incorporate techniques like dense learning signals, surprise minimization, exploration strategies, and blending
reinforcement and imitation learning.

#### 3. Scalable Training Infrastructure

Investigate scalable training approaches, including distributed reinforcement learning, student-teacher architectures,
and blending reinforcement learning with imitation learning, to enable efficient training of large-scale multi-agent
systems.

#### 4. Intelligence Evaluations for Gridworld Agents

Design and implement a comprehensive suite of intelligence evaluations for gridworld agents, covering navigation tasks,
maze solving, in-context learning, cooperation, and competition scenarios.

#### 5. DevOps and Tooling

Develop tools and infrastructure for efficient management, tracking, and deployment of experiments, such as cloud
cluster management, experiment tracking and visualization, and continuous integration and deployment pipelines.

This README provides only a brief overview of research explorations. Visit the
[research roadmap](https://github.com/Metta-AI/metta/blob/main/roadmap.md) for more details.

## Installation

### Quick Start

Clone the repository and run the setup:

```bash
git clone https://github.com/Metta-AI/metta.git
cd metta
./install.sh
```

The interactive setup will guide you through configuration:
- **Option 1**: Quick setup with defaults for external users (recommended for most users)
- **Option 2**: Quick setup with defaults for Softmax employees  
- **Option 3**: Full configuration wizard (customize everything)

After installation, you can use metta commands directly:

```bash
metta status       # Check component status
metta install      # Install additional components
metta configure    # Reconfigure for a different profile
```

### Installation Options

For automated/scripted setups:

```bash
# Quick setup with external defaults (no prompts)
./install.sh --profile=external --non-interactive

# Quick setup with Softmax defaults  
./install.sh --profile=softmax --non-interactive

# Fully automated setup (uses external defaults)
./install.sh --non-interactive
```

**Re-running setup**: The installer will always ask you to choose your profile during setup, ensuring proper configuration for new users.

## Configuration

Metta uses a unified profile-based configuration system with a single project-level configuration file, making it simple and discoverable for both individual developers and teams.

### Configuration File

The configuration file is located in your project root:
- **`config.yaml`** - Single configuration file in the repository root
- **Committed to version control** - Team shares the same base configuration
- **Profile-based** - Different user types (external/softmax) have appropriate defaults
- **Environment variable support** - For runtime overrides when needed

### Profile System

Metta supports different user profiles with appropriate defaults:

- **`external`** - For open source contributors and researchers
  - Basic W&B tracking enabled
  - Local storage for replays and checkpoints  
  - No cloud services or monitoring by default

- **`softmax`** - For Softmax team members
  - Team W&B workspace (softmax-ai/metta-internal)
  - S3 cloud storage for artifacts
  - Datadog monitoring enabled

### Interactive Configuration

Set up or modify your configuration:

```bash
metta configure              # Full interactive wizard
metta configure wandb        # Configure specific component
metta profile external      # Switch to external profile
metta profile softmax       # Switch to softmax profile
metta profiles              # List available profiles
```

The wizard will:
- Walk you through each service with intelligent defaults
- Ask profile-appropriate questions based on your selected profile
- Save settings to the project configuration file
- Allow component-specific customization

### Configuration File Structure

The project `config.yaml` uses a clean profile-based structure:

```yaml
# config.yaml (committed to repository)
# Default active profile - can be overridden with METTA_PROFILE env var
profile: external

profiles:
  # External Contributors/Researchers Profile
  # - Basic W&B tracking enabled for experiment logging
  # - Local storage for replays and checkpoints
  # - No cloud services or monitoring
  external:
    wandb:
      enabled: true
      # entity and project can be configured during setup or via 'metta configure wandb'
    observatory:
      enabled: false
    datadog:
      enabled: false

  # Softmax Team Internal Profile  
  # - Configured for internal team workflows
  # - Uses team W&B workspace and S3 storage
  # - Datadog monitoring enabled for production runs
  softmax:
    wandb:
      enabled: true
      entity: softmax-ai
      project: metta-internal
    observatory:
      enabled: false
    storage:
      s3_bucket: metta-softmax-storage
      aws_profile: softmax
      replay_dir: s3://metta-softmax-storage/replays/
      torch_profile_dir: s3://softmax-public/torch_traces/
    datadog:
      enabled: true
```

### Using Different Profiles

Switch profiles for different contexts:

```bash
# Switch active profile permanently
metta profile external
metta profile softmax

# Use profile for single command
METTA_PROFILE=softmax ./tools/run.py experiments.recipes.arena.train --args run=my_experiment

# Check current profile
metta profiles
```

### DevOps Integration

Export configuration for deployment tools and CI/CD:

```bash
# Shell export format (for sourcing)
metta export-env

# Export with specific profile
METTA_PROFILE=softmax metta export-env

# JSON format (for APIs)
metta export-env --format=json

# .env file format (for containers)  
metta export-env --format=file .env
```

### Configuration Priority

Metta uses a straightforward priority system:

1. **`--profile` CLI parameter** (highest priority)
2. **`METTA_PROFILE` environment variable**
3. **`profile` setting in config.yaml**
4. **`"external"` default** (lowest priority)

The system prioritizes profile-based configuration over environment variables, with environment variables serving as runtime overrides when needed.

## Usage

The repository contains command-line tools in the `tools/` directory.

### Run tasks with the runner

`run.py` is a script that kicks off tasks like training, evaluation, and visualization. The runner looks up the task,
builds its configuration, and runs it. The current available tasks are:

- **experiments.recipes.arena.train**: Train on the arena curriculum

  `./tools/run.py experiments.recipes.arena.train --args run=my_experiment`

- **experiments.recipes.navigation.train**: Train on the navigation curriculum

  `./tools/run.py experiments.recipes.navigation.train --args run=my_experiment`

- **experiments.recipes.arena.play**: Play in the browser

  `./tools/run.py experiments.recipes.arena.play`

- **experiments.recipes.arena.replay**: Replay a single episode from a saved policy

  `./tools/run.py experiments.recipes.arena.replay --overrides policy_uri=wandb://run/local.alice.1`

- **experiments.recipes.arena.evaluate**: Evaluate a policy on the arena eval suite

  `./tools/run.py experiments.recipes.arena.evaluate --args policy_uri=wandb://run/local.alice.1`

- Dry-run version, e.g. Print the resolved config without executing it

  `./tools/run.py experiments.recipes.arena.train --args run=my_experiment --dry-run`

### Runner arguments

Use the runner like this:

```bash
./tools/run.py <task_name> [--args key=value ...] [--overrides path.to.field=value ...] [--dry-run]
```

- `task_name`: a Python-style path to a task (for example, `experiments.recipes.arena.train`).
- `--args`: name=value pairs passed to the task function (these become constructor args of the Tool it returns).
  - Types: integers (`42`), floats (`0.1`), booleans (`true/false`), and strings.
  - Multiple args: add more pairs separated by spaces.
  - Example: `--args run=local.alice.1`
- `--overrides`: update fields inside the returned Tool configuration using dot paths.
  - Common fields: `system.device=cpu`, `wandb.enabled=false`, `trainer.total_timesteps=100000`,
    `trainer.rollout_workers=4`, `policy_uri=wandb://run/<name>` (for replay/eval).
  - Multiple overrides: add more pairs separated by spaces.
  - Example: `--overrides system.device=cpu wandb.enabled=false`
- `--dry-run`: print the fully-resolved configuration as JSON and exit without running.

Quick examples:

```bash
# Faster local run on CPU, less logging
./tools/run.py experiments.recipes.arena.train \
  --args run=local.alice.1 \
  --overrides system.device=cpu wandb.enabled=false trainer.total_timesteps=100000

# Evaluate a specific policy URI on the arena suite
./tools/run.py experiments.recipes.arena.evaluate --args policy_uri=wandb://run/local.alice.1
```

Tips:

- Strings with spaces: quote the value, for example `notes="my local run"`.
- Booleans are lowercase: `true` and `false`.
- If a value looks numeric but should be a string, wrap it in quotes (for example, `run="001"`).

### Defining your own runner tasks

A “task” is just a Python function (or class) that returns a Tool configuration. The runner loads it by name and runs
its `invoke()` method.

What you write:

- A function that returns a Tool, for example `TrainTool`, `SimTool`, `PlayTool`, or `ReplayTool`.
- Place it anywhere importable (for personal use, `experiments/user/<your_file>.py` is convenient).
- The function name becomes part of the task name you run.

Minimal example:

```python
# experiments/user/my_tasks.py
from metta.mettagrid.config.envs import make_arena
from metta.rl.trainer_config import EvaluationConfig, TrainerConfig
from metta.sim.simulation_config import SimulationConfig
from metta.tools.train import TrainTool

def my_train(run: str = "local.me.1") -> TrainTool:
    trainer = TrainerConfig(
        evaluation=EvaluationConfig(
            simulations=[SimulationConfig(name="arena/basic", env=make_arena(num_agents=4))]
        )
    )
    return TrainTool(trainer=trainer, run=run)
```

Run your task:

```bash
./tools/run.py experiments.user.my_tasks.my_train --args run=local.me.2 \
  --overrides system.device=cpu wandb.enabled=false
```

Notes:

- Tasks can also be Tool classes (subclasses of `metta.common.config.tool.Tool`). The runner will construct them with
  `--args` and then apply `--overrides`.
- Use `--dry-run` while developing to see the exact configuration your task produces.

### Setting up Weights & Biases for Personal Use

To use WandB with your personal account:

1. Get your WandB API key from [wandb.ai](https://wandb.ai) (click your profile → API keys)
2. Add it to your `~/.netrc` file:
   ```
   machine api.wandb.ai
     login user
     password YOUR_API_KEY_HERE
   ```
3. Configure WandB using the interactive wizard:
   ```bash
   metta configure wandb
   ```
   This will prompt you for your entity and project settings, and save them to the project config file.

Now you can run training with WandB enabled (it's enabled by default in both profiles):

```bash
./tools/run.py experiments.recipes.arena.train --args run=local.<your_name>.123
```

If you need to override the entity for a specific run:
```bash
./tools/run.py experiments.recipes.arena.train --args run=my_experiment --overrides wandb.entity=<your_user>
```

## Visualizing a Model

### Mettascope: in-browser viewer

Mettascope allows you to run and view episodes in the environment you specify. It goes beyond just spectator mode, and
allows taking over an agent and controlling it manually.

For more information, see [./mettascope/README.md](./mettascope/README.md).

#### Run the interactive simulation

```bash
./tools/run.py experiments.recipes.arena.play
```

Optional overrides:

- `policy_uri=<path>`: Use a specific policy for NPC agents.
  - Local checkpoints: `file://./train_dir/<run>/checkpoints`
  - WandB artifacts: `wandb://run/<run_name>`

### Replay a single episode

```
./tools/run.py experiments.recipes.arena.replay --overrides policy_uri=wandb://run/local.alice.1
```

### Evaluating a Model

When you run training, if you have WandB enabled, then you will be able to see in your WandB run page results for the
eval suites.

However, this will not apply for anything trained before April 8th.

#### Post Hoc Evaluation

If you want to run evaluation post-training to compare different policies, you can do the following:

Evaluate a policy against the arena eval suite:

```
./tools/run.py experiments.recipes.arena.evaluate --args policy_uri=wandb://run/local.alice.1
```

Evaluate on the navigation eval suite (provide the policy URI):

```bash
./tools/run.py experiments.recipes.navigation.eval --args policy_uri=wandb://run/local.alice.1
```

### Specifying your agent architecture

#### Configuring a MettaAgent

This repo implements a `MettaAgent` policy class. The underlying network is parameterized by config files in
`configs/agent` (with `configs/agent/fast.yaml` used by default). See `configs/agent/reference_design.yaml` for an
explanation of the config structure, and [this wiki section](https://deepwiki.com/Metta-AI/metta/6-agent-architecture)
for further documentation.

To use `MettaAgent` with a non-default architecture config:

- (Optional): Create your own configuration file, e.g. `configs/agent/my_agent.yaml`.
- Run with the configuration file of your choice:
  ```bash
  ./tools/run.py experiments.recipes.arena.train --overrides policy_architecture.agent_config=my_agent
  ```

#### Defining your own PyTorch agent

We support agent architectures without using the MettaAgent system:

- Implement your agent class under `metta/agent/src/metta/agent/pytorch/my_agent.py`. See
  `metta/agent/src/metta/agent/pytorch/fast.py` for an example.
- Register it in `metta/agent/src/metta/agent/pytorch/agent_mapper.py` by adding an entry to `agent_classes` with a key
  name (e.g., `"my_agent"`).
- Select it at runtime using the runner and an override on the agent config name:
  ```bash
  ./tools/run.py experiments.recipes.arena.train --overrides policy_architecture.name=pytorch/my_agent
  ```

Further updates to support bringing your own agent are coming soon.

## Development Setup

To run the style checks and tests locally:

```bash
ruff format
ruff check
pyright metta  # optional, some stubs are missing
pytest
```

### CLI cheat sheet

| Task                        | Command                                                                                                |
| --------------------------- | ------------------------------------------------------------------------------------------------------ |
| Train (arena)               | `./tools/run.py experiments.recipes.arena.train --args run=my_experiment`                              |
| Train (navigation)          | `./tools/run.py experiments.recipes.navigation.train --args run=my_experiment`                         |
| Play (browser)              | `./tools/run.py experiments.recipes.arena.play`                                                        |
| Replay (policy)             | `./tools/run.py experiments.recipes.arena.replay --overrides policy_uri=wandb://run/local.alice.1`     |
| Evaluate (arena)            | `./tools/run.py experiments.recipes.arena.evaluate --args policy_uri=wandb://run/local.alice.1`        |
| Evaluate (navigation suite) | `./tools/run.py experiments.recipes.navigation.eval --args policy_uri=wandb://run/local.alice.1` |
| Dry-run (print config)      | `./tools/run.py experiments.recipes.arena.train --args run=my_experiment --dry-run`                    |

Running these commands mirrors our CI configuration and helps keep the codebase consistent.
