# Metta AI Repository

## Overview

Metta AI is a reinforcement learning codebase focusing on the emergence of cooperation and alignment in multi-agent AI systems. It's a simulation environment (game) designed to train AI agents capable of meta-learning general intelligence through social dynamics and kinship structures.

## Architecture

This is a multi-language repository with the following main components:

### Core Components
- **Python (Primary)**: Main framework and RL training infrastructure
- **C++**: High-performance simulation engine (mettagrid)
- **TypeScript/JavaScript**: Web-based visualization and map editing tools

### Key Modules
- `metta/`: Core Python package containing:
  - `agent/`: Agent architectures and policy management
  - `rl/`: Reinforcement learning components (trainers, losses, experience)
  - `sim/`: Simulation environment and configuration
  - `map/`: Map generation and loading utilities
  - `eval/`: Evaluation and analysis tools
  - `util/`: Utility functions and helpers

- `mettagrid/`: C++ simulation engine with Python bindings
- `app_backend/`: FastAPI backend for web services
- `mettascope/`: Web-based replay viewer
- `mettamap/`: Next.js map editor interface
- `observatory/`: Visualization dashboard

## Installation

### Prerequisites
- Python 3.11.7 (exact version required)
- uv package manager

### Setup
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Optional: Setup development environment (requires team permissions)
./devops/setup_dev.sh
```

## Running the Code

### Training a Model
```bash
./tools/train.py run=my_experiment +hardware=macbook wandb=off
```

### Interactive Simulation
```bash
./tools/play.py run=my_experiment +hardware=macbook wandb=off
```

### Terminal Simulation
```bash
./tools/renderer.py run=demo_obstacles \
renderer_job.environment.uri="configs/env/mettagrid/maps/debug/simple_obstacles.map"
```

### Evaluation
```bash
./tools/sim.py \
    sim=navigation \
    run=navigation101 \
    policy_uri=wandb://run/YOUR_POLICY_URI \
    sim_job.stats_db_uri=wandb://stats/navigation_db \
    device=cpu
```

### Dashboard
```bash
./tools/dashboard.py +eval_db_uri=wandb://stats/navigation_db run=navigation_db
```

## Development

### Code Quality
```bash
ruff format          # Format code
ruff check           # Lint code
pyright metta        # Type checking (optional)
pytest               # Run tests
```

### Testing
- Test framework: pytest
- Coverage: pytest-cov
- Test location: `tests/` directory
- Configuration: `pyproject.toml`

### Key Features
- Multi-agent gridworld environment
- Kinship and social dynamics simulation
- Reward sharing mechanisms
- Energy management and resource systems
- Combat and cooperation mechanics
- Map generation and procedural content
- Distributed training infrastructure
- Web-based visualization tools

## Project Structure
```
metta/
├── metta/              # Core Python package
├── mettagrid/          # C++ simulation engine
├── tests/              # Test suite
├── tools/              # CLI tools and scripts
├── configs/            # Configuration files
├── app_backend/        # FastAPI backend
├── mettascope/         # Replay viewer
├── mettamap/           # Map editor
├── observatory/        # Dashboard
├── devops/             # Infrastructure and deployment
└── docs/               # Documentation
```

## Research Focus
The project investigates how social dynamics (kinship, mate selection) can lead to:
- Emergence of cooperation in AI systems
- Development of general intelligence
- AI alignment through social mechanisms
- Complex multi-agent behaviors

This is an active research project exploring the hypothesis that "love" (social bonds) plays a crucial role in developing cooperative AGI.