# Metta AI Repository

## Overview

Metta AI is an open-source research project investigating the emergence of cooperation and alignment in multi-agent AI
systems. The project creates a model organism for complex multi-agent gridworld environments to study the impact of
social dynamics, such as kinship and mate selection, on learning and cooperative behaviors of AI agents.

The core hypothesis is that social dynamics, akin to love in biological systems, play a crucial role in the development
of cooperative AGI and AI alignment. The project introduces a novel reward-sharing mechanism mimicking familial bonds
and mate selection, allowing researchers to observe the evolution of complex social behaviors and cooperation among AI
agents.

## Repository Structure

The repository is organized into several key components:

- **metta/**: Core library containing agent architectures, evaluation tools, simulation components, and utilities
- **mettagrid/**: Grid-based environment implementation
- **mettamap/**: Map visualization and editing tools
- **mettascope/**: Visualization and analysis tools for agent behaviors
- **tools/**: Command-line tools for training, evaluation, and visualization
- **configs/**: Configuration files for environments, agents, and training
- **scenes/**: Pre-defined environment scenes and maps
- **tests/**: Test suite for the codebase

## Installation

1. Install uv (a fast Python package installer and resolver):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Clone the repository and install dependencies:

```bash
git clone https://github.com/Metta-AI/metta.git
cd metta
uv sync
```

Note: The project requires Python 3.11.7 specifically, as specified in the pyproject.toml file.

## Running the Code

### Training a Model

To train a model:

```bash
./tools/train.py run=my_experiment wandb=off
```

Parameters:

- `run`: Names your experiment and controls where checkpoints are saved under `train_dir/<run>`
- `+hardware=<preset>`: Tunes the trainer for your machine (options include macbook, desktop, etc.)
- `+user=<n>`: Loads defaults from `configs/user/<n>.yaml`
- `wandb=off`: Disables Weights & Biases logging if you don't have access

### Visualizing a Model

To run the interactive simulation:

```bash
./tools/play.py run=my_experiment wandb=off
```

This launches a human-controlled session using the same configuration flags as training. It's useful for quickly testing
maps or policies on your local hardware.

To run the terminal simulation:

```bash
./tools/renderer.py run=demo_obstacles \
renderer_job.environment.root.params.uri="configs/env/mettagrid/maps/debug/simple_obstacles.map"
```

### Evaluating a Model

When you run training with WandB enabled, you'll see results for the eval suites in your WandB run page.

For post-training evaluation to compare different policies:

1. Add your policy to the existing navigation evals DB:

```bash
./tools/sim.py \
    sim=navigation \
    run=navigation101 \
    policy_uri=wandb://run/YOUR_POLICY_URI \
    sim_job.stats_db_uri=wandb://stats/navigation_db \
    device=cpu
```

2. View the results in a heatmap along with other policies in the database:

```bash
./tools/dashboard.py +eval_db_uri=wandb://stats/navigation_db run=navigation_db ++dashboard.output_path=s3://softmax-public/policydash/navigation.html
```

## Development Setup

To run style checks and tests locally:

```bash
ruff format
ruff check
pyright metta  # optional, some stubs are missing
pytest
```

These commands mirror the CI configuration and help keep the codebase consistent.

## Key Features

1. **Multi-agent Gridworld Environment**: A flexible environment where agents can interact, compete, and cooperate
2. **Resource Management**: Agents harvest diamonds, convert them to energy, and use energy for various actions
3. **Combat and Defense**: Agents can attack others or defend themselves with shields
4. **Cooperation Mechanisms**: Agents can share energy or resources and use markers to communicate
5. **Kinship Structures**: Flexible kinship structure simulating relationships from close kin to strangers
6. **Visualization Tools**: Tools for visualizing agent behaviors and environment dynamics

## Research Applications

The project is designed for research in:

1. **Environment Development**: Creating rich gridworld environments with complex dynamics
2. **Agent Architecture Research**: Incorporating techniques like dense learning signals and exploration strategies
3. **Scalable Training Infrastructure**: Investigating distributed reinforcement learning approaches
4. **Intelligence Evaluations**: Designing comprehensive intelligence tests for gridworld agents
5. **Cooperation and Alignment**: Studying the emergence of cooperative behaviors in multi-agent systems

## Community and Resources

- **Discord**: https://discord.gg/mQzrgwqmwy
- **Short (5m) Talk**: https://www.youtube.com/watch?v=bt6hV73VA8I
- **Talk**: https://foresight.org/summary/david-bloomin-metta-learning-love-is-all-you-need/
- **Interactive Demo**:
  https://metta-ai.github.io/metta/?replayUrl=https%3A%2F%2Fsoftmax-public.s3.us-east-1.amazonaws.com%2Freplays%2Fandre_pufferbox_33%2Freplay.77200.json.z&play=true
