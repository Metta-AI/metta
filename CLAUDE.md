# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Metta AI is a reinforcement learning project focusing on the emergence of cooperation and alignment in multi-agent AI systems. It creates a model organism for complex multi-agent gridworld environments to study the impact of social dynamics (like kinship and mate selection) on learning and cooperative behaviors.

The codebase consists of:
- `metta/`: Core Python implementation for agents, maps, RL algorithms, simulation
- `mettagrid/`: C++/Python grid environment implementation
- `mettascope/`: Visualization and replay tools

## Development Environment Setup

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Run setup script (creates virtual environment automatically)
./devops/setup_build.sh

# Rebuild only mettagrid component
./devops/build_mettagrid.sh
```

## Common Commands

### Training and Simulation

```bash
# Train a model
python -m tools.train run=my_experiment +hardware=macbook wandb=off

# Run evaluation
python -m tools.sim run=my_experiment +hardware=macbook wandb=off

# Run interactive simulation
python -m tools.play run=my_experiment +hardware=macbook wandb=off
```

### Evaluation

```bash
# Add a policy to the navigation evals database
python -m tools.sim eval=navigation run=RUN_NAME eval.policy_uri=POLICY_URI +eval_db_uri=wandb://artifacts/navigation_db

# Analyze results with heatmap
python -m tools.analyze run=analyze +eval_db_uri=wandb://artifacts/navigation_db analyzer.policy_uri=POLICY_URI
```

### Code Quality

```bash
# Run all tests with coverage
pytest --cov=mettagrid --cov-report=term-missing

# Run linting with Ruff
ruff check .

# Auto-fix Ruff errors with Claude (requires ANTHROPIC_API_KEY)
python -m devops.tools.auto_ruff_fix path/to/file

# Format shell scripts
./devops/tools/format_sh.sh
```

### Building

```bash
# Clean build artifacts
make clean

# Build from setup.py
make build

# Build and install
make install

# Run tests
make test

# Full clean, install, and test
make all
```

## Code Architecture

### Agent System

- Each agent has a policy with action spaces and observation spaces
- Policies are stored in `PolicyStore` and managed by `MettaAgent`
- Agent architecture is designed to be adaptable to new game rules and environments
- Neural components can be mixed and matched via configuration

### Environment System

- Gridworld environments with agents, resources, and interaction rules
- Procedural world generation with customizable configurations
- Various environment types with different dynamics and challenges
- Support for different kinship schemes and mate selection mechanisms

### Training Infrastructure

- Distributed reinforcement learning with multi-GPU support
- Integration with Weights & Biases for experiment tracking
- Scalable architecture for training large-scale multi-agent systems
- Support for curriculum learning and knowledge distillation

### Evaluation System

- Comprehensive suite of intelligence evaluations
- Navigation tasks, maze solving, in-context learning
- Cooperation and competition metrics
- Support for tracking and comparing multiple policies

## Configuration System

The project uses OmegaConf for configuration, with config files organized in `configs/`:

- `agent/`: Agent architecture configurations
- `trainer/`: Training configurations
- `sim/`: Simulation configurations
- `hardware/`: Hardware-specific settings
- `user/`: User-specific configurations

## Testing Philosophy

- Tests should be independent and idempotent
- Tests should be focused on testing one thing
- Tests should cover edge cases and boundary conditions
- Tests are organized in the `tests/` directory, mirroring the project structure