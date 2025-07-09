# Metta Repository Organization Plan - Flattened Structure

## Overview

This document outlines a flattened organization structure for the Metta monorepo, emphasizing simplicity and clear package boundaries.

### Goals of the Flattened Structure

Our flattened package structure achieves these key goals:

#### Technical Goals

1. **Independent Packages**: Each major component is its own Python package with distinct namespaces
2. **Simple Imports**: Direct package names without deep nesting (e.g., `import mettagrid`, `from cogworks import train`)
3. **Clear Boundaries**: Each package has a focused purpose and minimal dependencies

#### Organizational Goals

4. **Easy Publishing**: Any package can be published to PyPI independently with minimal effort
5. **Shallow Hierarchy**: Maximum 2 levels deep for any module path
6. **Developer Ergonomics**: Simple, predictable structure that's easy to navigate

### Import Philosophy

Rather than forcing everything under a `metta.*` namespace, we embrace distinct package identities:

```python
# Clear, focused imports
import mettagrid                    # Environment package
from cogworks import train          # RL training
from cogworks.rl import trainer     # RL trainer module
from mettacommon import logger      # Shared utilities
from gridworks import MapEditor     # Map editing tools
```

## Core Principles

1. **Flat is Better**: Avoid nested src/ directories and deep hierarchies
2. **Package Independence**: Each package can be `pip install`ed separately
3. **Clear Namespaces**: Each package owns its namespace (no forced metta.* prefix)
4. **Minimal Shared Code**: Only truly universal utilities go in common
5. **Direct Imports**: Prefer `from package import module` over `from package.subpackage.submodule import thing`

## Package Structure

### Current Structure → Proposed Structure

Here's how the existing repository maps to the new flattened structure:

```
# CURRENT STRUCTURE                 → PROPOSED STRUCTURE
metta/                              → Softmax/
├── metta/                          → cogworks/
│   ├── __init__.py                   ├── __init__.py
│   ├── api.py                        ├── api.py
│   ├── rl/                           ├── rl/
│   ├── eval/                         ├── eval/
│   ├── sweep/                        ├── sweep/
│   ├── map/                          ├── mapgen/
│   ├── sim/                          ├── sim/
│   └── setup/                        └── setup/
├── agent/                          → (merged into cogworks/agent/)
├── common/                         → mettacommon/
├── mettagrid/                      → mettagrid/ (unchanged)
├── mettascope/                     → mettascope/ (unchanged)
├── observatory/                    → observatory/ (unchanged)
├── studio/                         → gridworks/
├── app_backend/                    → (split into respective apps)
├── tools/                          → tools/ (unchanged)
├── configs/                        → configs/ (unchanged)
├── scenes/                         → scenes/ (unchanged)
├── recipes/                        → recipes/ (moved to cogworks/recipes/)
├── tests/                          → (distributed to packages)
├── docs/                           → docs/ (unchanged)
├── devops/                         → devops/ (unchanged)
└── wandb_carbs/                    → (integrated or removed)
```

### Detailed New Structure

```
Softmax/
├── pyproject.toml              # Workspace configuration
├── uv.lock                     # Unified lock file
│
├── cogworks/                   # RL training framework (from metta/ + agent/)
│   ├── pyproject.toml          # name = "cogworks"
│   ├── __init__.py
│   ├── api.py                  # Main training APIs (from metta/api.py)
│   ├── agent/                  # Agent/policy code (from agent/src/metta/agent/)
│   │   ├── metta_agent.py      # Main agent class
│   │   ├── policy_store.py     # Policy storage
│   │   ├── policy_cache.py     # Policy caching
│   │   ├── policy_record.py    # Policy records
│   │   ├── policy_metadata.py  # Policy metadata
│   │   ├── policy_state.py     # Policy state
│   │   ├── lib/                # Agent libraries (LSTM, etc.)
│   │   ├── external/           # External integrations
│   │   └── util/               # Agent utilities
│   ├── rl/                     # RL algorithms (from metta/rl/)
│   │   ├── trainer.py          # Main trainer
│   │   ├── trainer_config.py   # Trainer configuration
│   │   ├── functions.py        # Functional training utilities
│   │   ├── experience.py       # Experience buffer
│   │   ├── losses.py           # Loss functions
│   │   ├── policy.py           # Policy utilities
│   │   ├── kickstarter.py      # Kickstarter training
│   │   ├── vecenv.py           # Vectorized environments
│   │   ├── fast_gae.cpp        # C++ GAE implementation
│   │   └── fast_gae/           # Fast GAE module
│   ├── eval/                   # Evaluation tools (from metta/eval/)
│   │   ├── analysis.py         # Analysis tools
│   │   ├── dashboard_data.py   # Dashboard data generation
│   │   ├── eval_stats_db.py    # Evaluation statistics DB
│   │   └── analysis_config.py  # Analysis configuration
│   ├── sweep/                  # Hyperparameter sweeping (from metta/sweep/)
│   │   ├── protein.py          # Main sweep engine
│   │   ├── protein_metta.py    # Metta-specific sweep
│   │   └── protein_wandb.py    # WandB sweep integration
│   ├── sim/                    # Simulation management (from metta/sim/)
│   │   ├── simulation.py       # Core simulation
│   │   ├── simulation_stats_db.py  # Simulation stats DB
│   │   ├── simulation_suite.py # Simulation suite
│   │   ├── simulation_config.py # Simulation config
│   │   └── map_preview.py      # Map preview utilities
│   ├── mapgen/                 # Map generation (from metta/map/)
│   │   ├── mapgen.py           # Map generator
│   │   ├── scene.py            # Scene management
│   │   ├── types.py            # Map types
│   │   ├── load.py             # Map loading
│   │   ├── load_random.py      # Random map loading
│   │   ├── config.py           # Map config
│   │   ├── random/             # Random generation algorithms
│   │   ├── scenes/             # Scene implementations
│   │   └── utils/              # Map utilities
│   ├── setup/                  # Setup and installation tools (from metta/setup/)
│   ├── recipes/                # Example scripts (from root recipes/)
│   └── tests/                  # All RL/agent tests
│
├── mettagrid/                  # C++/Python environment (flattened structure)
│   ├── pyproject.toml          # name = "mettagrid"
│   ├── CMakeLists.txt
│   ├── __init__.py
│   ├── mettagrid_env.py        # Main environment class
│   ├── mettagrid_config.py     # Environment configuration
│   ├── char_encoder.py         # Character encoding
│   ├── gym_wrapper.py          # Gym compatibility wrapper
│   ├── level_builder.py        # Level building utilities
│   ├── replay_writer.py        # Replay recording
│   ├── stats_writer.py         # Stats recording
│   ├── episode_stats_db.py     # Episode statistics
│   ├── mettagrid_c.cpp         # C++ bindings
│   ├── mettagrid_c.pyi         # Type stubs
│   ├── grid.hpp                # Core grid implementation
│   ├── action_handler.hpp      # Action handling
│   ├── observation_encoder.hpp # Observation encoding
│   ├── curriculum/             # Curriculum definitions
│   ├── actions/                # Action implementations
│   ├── objects/                # Game objects
│   ├── renderer/               # Rendering utilities
│   ├── room/                   # Room generation
│   ├── util/                   # C++ utilities
│   ├── configs/                # Environment configs
│   ├── tests/                  # C++ and Python tests
│   └── benchmarks/             # Performance benchmarks
│
├── mettacommon/                # Minimal shared utilities (from common/)
│   ├── pyproject.toml          # name = "mettacommon"
│   ├── __init__.py
│   ├── util/                   # Common utilities
│   │   ├── config.py           # Config management
│   │   ├── resolvers.py        # Hydra resolvers
│   │   ├── logging_helpers.py  # Logging setup
│   │   ├── system_monitor.py   # System monitoring
│   │   ├── runtime_configuration.py # Runtime config
│   │   ├── fs.py               # Filesystem utilities
│   │   ├── git.py              # Git utilities
│   │   ├── heartbeat.py        # Process heartbeat
│   │   ├── cli.py              # CLI helpers
│   │   ├── colorama.py         # Color output
│   │   ├── datastruct.py       # Data structures
│   │   └── tracing.py          # Tracing utilities
│   ├── profiling/              # Performance monitoring
│   │   ├── memory_monitor.py   # Memory monitoring
│   │   └── stopwatch.py        # Timing utilities
│   ├── wandb/                  # WandB integration
│   └── tests/                  # Common tests
│
├── gridworks/                  # Map editor and studio (from studio/)
│   ├── pyproject.toml          # name = "gridworks"
│   ├── package.json
│   ├── next.config.ts
│   ├── next-env.d.ts
│   ├── tailwind.config.ts
│   ├── tsconfig.json
│   ├── server/                 # Python backend (if needed)
│   ├── src/                    # TypeScript/React frontend
│   │   ├── app/                # Next.js app directory
│   │   ├── components/         # React components
│   │   ├── hooks/              # React hooks
│   │   ├── icons/              # Icon components
│   │   ├── lib/                # Utilities
│   │   └── server/             # Server actions
│   ├── public/                 # Static assets
│   └── tests/                  # Frontend tests
│
├── observatory/                # Production monitoring (mostly unchanged)
│   ├── pyproject.toml          # name = "observatory"
│   ├── package.json
│   ├── vite.config.ts
│   ├── index.html
│   ├── src/                    # React frontend
│   │   ├── App.tsx
│   │   ├── Dashboard.tsx
│   │   ├── components/
│   │   └── config.ts
│   ├── api/                    # Backend API (from app_backend/)
│   │   ├── __init__.py
│   │   ├── endpoints.py
│   │   └── models.py
│   ├── docker/                 # Deployment configs
│   │   └── Dockerfile
│   └── tests/
│
├── mettascope/                 # Replay viewer (mostly unchanged)
│   ├── pyproject.toml          # name = "mettascope"
│   ├── package.json
│   ├── vite.config.mts
│   ├── index.html
│   ├── __init__.py
│   ├── src/                    # TypeScript frontend
│   │   ├── main.ts
│   │   ├── renderer.ts
│   │   ├── replay.ts
│   │   └── ui.ts
│   ├── data/                   # Assets
│   │   ├── atlas/              # Sprite atlases
│   │   ├── fonts/              # Font files
│   │   ├── ui/                 # UI assets
│   │   └── view/               # View assets
│   ├── tools/                  # Build tools
│   │   ├── gen_atlas.py
│   │   └── gen_html.py
│   └── tests/
│
├── tools/                      # Standalone entry scripts (unchanged)
│   ├── train.py                # Main training script
│   ├── sweep_init.py           # Sweep initialization
│   ├── sweep_eval.py           # Sweep evaluation
│   ├── sweep_config_utils.py   # Sweep config utilities
│   ├── play.py                 # Interactive play
│   ├── sim.py                  # Run simulations
│   ├── replay.py               # Generate replays
│   ├── analyze.py              # Analysis tools
│   ├── dashboard.py            # Dashboard launcher
│   ├── renderer.py             # Render utilities
│   ├── validate_config.py      # Config validation
│   ├── stats_duckdb_cli.py     # Stats DB CLI
│   ├── upload_map_imgs.py      # Map image upload
│   ├── autotune.py             # Auto-tuning script
│   ├── dump_src.py             # Source dumping utility
│   └── map/                    # Map generation tools
│       ├── gen.py
│       ├── gen_scene.py
│       └── normalize_ascii_map.py
│
├── configs/                    # Hydra configurations (unchanged)
│   ├── agent/
│   ├── env/
│   │   └── mettagrid/
│   ├── trainer/
│   ├── hardware/
│   ├── sim/
│   ├── sweep/
│   ├── user/
│   ├── wandb/
│   └── *.yaml                  # Root config files
│
├── scenes/                     # Map/scene definitions (unchanged)
│   ├── convchain/
│   ├── dcss/
│   ├── test/
│   └── wfc/
│
├── docs/                       # Documentation (unchanged)
│   ├── api.md
│   ├── mapgen.md
│   ├── wandb/
│   │   └── metrics/
│   └── workflows/
│
├── devops/                     # Infrastructure and tooling (unchanged)
│   ├── aws/
│   ├── charts/
│   ├── docker/
│   ├── git-hooks/
│   ├── macos/
│   ├── skypilot/
│   ├── tf/
│   ├── tools/
│   └── wandb/
│
├── recipes/                    # Recipe scripts (if not moved to cogworks)
├── checkpoints/                # Model checkpoints (gitignored)
├── wandb/                      # WandB runs (gitignored)
└── .github/                    # GitHub workflows
```

## Package Details

### Training Framework (`cogworks`)

**Namespace**: `cogworks`
**Purpose**: Core RL training functionality

```python
# Example imports
from cogworks import api
from cogworks.rl import trainer, experience
from cogworks.agent import metta_agent, policy_store
from cogworks.sweep import protein
from cogworks.sim import simulation
```

**Why "cogworks"?**: Distinct identity for our RL framework, cognitive training workbench.

### Environment Engine (`mettagrid`)

**Namespace**: `mettagrid`
**Purpose**: High-performance grid environment

```python
# Example imports
import mettagrid
from mettagrid import mettagrid_env
from mettagrid.curriculum import NavigationCurriculum
from mettagrid import char_encoder, gym_wrapper
```

**Independence**: Can be installed standalone for researchers who just want the environment.

### Shared Utilities (`mettacommon`)

**Namespace**: `mettacommon`
**Purpose**: Minimal truly shared code

```python
# Example imports
from mettacommon import setup_logging, profile
from mettacommon.config import load_config
```

**Why "mettacommon"?**: Matches the single-word pattern of `mettagrid`. Simple, consistent naming across our core packages.

**Scope**: Only utilities needed by 2+ packages. If only web apps use it, it goes in a web-specific package.

### Map Editor (`gridworks`)

**Namespace**: `gridworks`
**Purpose**: Map creation and testing studio

Features both Python backend and TypeScript frontend in one package.

### Monitoring (`observatory`)

**Namespace**: `observatory`
**Purpose**: Production experiment tracking

Self-contained web application with its own backend.

### Replay Viewer (`mettascope`)

**Namespace**: `mettascope`
**Purpose**: Local replay analysis

Lightweight viewer that can run standalone.

## Entry Points

The `tools/` directory contains standalone scripts that compose functionality:

```python
# tools/train.py
#!/usr/bin/env python
"""Direct training without Hydra complexity."""
from cogworks import api
from mettagrid import mettagrid_env

if __name__ == "__main__":
    env = mettagrid_env.MettaGridPufferEnv(...)
    # Training logic here
```

## Dependency Graph

```mermaid
graph TD
    A[mettacommon] --> B[mettagrid]
    A --> C[cogworks]
    B --> C
    A --> D[gridworks]
    A --> E[observatory]
    A --> F[mettascope]
```

Rules:
1. `mettacommon` has no dependencies
2. `mettagrid` only depends on `mettacommon`
3. `cogworks` depends on `mettagrid` and `mettacommon`
4. Web apps depend only on `mettacommon` (not on cogworks)

## Installation Examples

```bash
# Just the environment
pip install mettagrid

# Training framework (includes mettagrid)
pip install cogworks

# Everything
pip install cogworks[all]

# Development
uv sync  # Installs all workspace packages
```

## Migration from Current Structure

### Phase 1: Flatten Hierarchy
- Remove unnecessary src/ directories
- Consolidate nested packages

### Phase 2: Namespace Migration
- Move from `metta.*` imports to direct package imports
- Update all import statements

### Phase 3: Package Independence
- Ensure each package has complete pyproject.toml
- Test independent installation

## Benefits of Flat Structure

1. **Discoverability**: Easy to see what's available at a glance
2. **Simple Imports**: No deep nesting to remember
3. **Tool Friendly**: IDEs and tools work better with flatter structures
4. **Beginner Friendly**: Lower cognitive load for new developers
5. **Flexible Publishing**: Any package can be extracted and published

## Package Publishing Strategy

Each package maintains PyPI readiness:

```toml
# mettagrid/pyproject.toml
[project]
name = "mettagrid"
version = "0.1.0"
description = "High-performance grid environments"
dependencies = ["mettacommon>=0.1.0"]

# cogworks/pyproject.toml
[project]
name = "cogworks"
version = "0.1.0"
description = "RL training framework"
dependencies = ["mettagrid>=0.1.0", "mettacommon>=0.1.0"]
```

This allows:
- Independent version management
- Clear dependency relationships
- Easy extraction for external use

## Future Considerations

This flattened structure provides flexibility for:
- Renaming packages as they mature
- Extracting packages to separate repos if needed
- Converting to a fuller microservice architecture
- Supporting both research and production use cases

The key is starting simple and flat, then adding structure only where truly needed.

## Development Workflow

All packages remain in the monorepo with:

- Shared tooling configuration (ruff, pyright, pytest)
- Unified CI/CD pipeline
- Single `uv.lock` for consistent dependencies
- Common development environment
- Shared git hooks and code quality standards

### Local Development

```bash
# Clone and setup
git clone https://github.com/metta-ai/metta
cd metta
uv sync

# Work on specific package
cd cogworks
uv run pytest

# Run from anywhere
uv run python tools/train.py
```

## Testing Strategy

Each package maintains its own focused test suite:

```
cogworks/
├── tests/
│   ├── test_trainer.py
│   ├── test_losses.py
│   └── rl/
│       └── test_ppo.py

mettagrid/
├── tests/
│   ├── test_env.py
│   ├── test_curriculum.py
│   └── cpp/
│       └── test_grid.cpp
```

### Testing Principles

1. **Unit tests stay with package**: Each package tests its own code
2. **Integration tests in tools/**: Cross-package tests live at the root
3. **Shared test utilities**: `mettacommon.testing` provides fixtures
4. **Fast feedback**: Package tests should run in <30 seconds

## What Goes Where - Detailed Examples

### cogworks/ - RL Training Framework

```
cogworks/
├── __init__.py
├── api.py              # Environment, Agent base classes
├── agent/              # From agent/src/metta/agent/
│   ├── metta_agent.py  # Main agent implementation
│   ├── policy_store.py # Policy storage and loading
│   ├── policy_cache.py # Policy caching layer
│   └── lib/            # Agent modules (LSTM, etc.)
├── rl/
│   ├── trainer.py      # Main training loop
│   ├── trainer_config.py # Configuration classes
│   ├── functions.py    # Functional training utilities
│   ├── experience.py   # Experience buffer
│   ├── losses.py       # Loss functions
│   ├── policy.py       # Policy utilities
│   └── kickstarter.py  # Kickstarter training
├── eval/
│   ├── analysis.py     # Analysis tools
│   ├── dashboard_data.py # Dashboard data generation
│   └── eval_stats_db.py # Evaluation database
├── sweep/
│   ├── protein.py      # Main sweep orchestration
│   ├── protein_metta.py # Metta-specific sweep
│   └── protein_wandb.py # WandB integration
├── sim/
│   ├── simulation.py   # Core simulation logic
│   ├── simulation_stats_db.py # Stats database
│   └── simulation_suite.py # Simulation suites
└── recipes/
    ├── basic_training.py # Simple training example
    └── distributed.py   # Multi-GPU example
```

### mettagrid/ - Environment Package

```
mettagrid/
├── __init__.py
├── mettagrid_env.py    # Main MettaGridPufferEnv class
├── mettagrid_config.py # Environment configuration
├── char_encoder.py     # Character encoding for observations
├── gym_wrapper.py      # Gym compatibility wrapper
├── level_builder.py    # Level generation utilities
├── replay_writer.py    # Replay recording
├── stats_writer.py     # Statistics collection
├── episode_stats_db.py # Episode statistics database
├── curriculum/
│   ├── __init__.py
│   ├── base.py         # Curriculum interface
│   ├── navigation.py   # Navigation tasks
│   └── configs/        # Task YAML files
├── cpp/                # C++ source files
│   ├── grid.cpp        # Core grid logic
│   ├── entities.cpp    # Agents and objects
│   └── physics.cpp     # Movement and collisions
├── bindings/
│   └── mettagrid_c.cpp # PyBind11 wrapper
└── assets/
    └── sprites/        # Visual assets
```

**Independence**: Can be installed standalone for researchers who just want the environment.

### mettacommon/ - Truly Shared Code

```
mettacommon/
├── __init__.py
├── util/
│   ├── config.py           # Config loading utilities
│   ├── resolvers.py        # Hydra resolvers
│   ├── logging_helpers.py  # Logging configuration
│   ├── system_monitor.py   # System monitoring
│   ├── fs.py               # Filesystem utilities
│   └── git.py              # Git utilities
├── profiling/
│   ├── memory_monitor.py   # Memory profiling
│   └── stopwatch.py        # Performance timing
└── wandb/                  # WandB utilities
```

**Rule of thumb**: If it's used by only one package, it doesn't belong here.

## Documentation Strategy

### Three Levels of Documentation

1. **README.md** - Quick start and overview (per package)
2. **docs/** - Comprehensive guides and tutorials (root level)
3. **DESIGN.md** - Architecture decisions (per complex package)

### Package Documentation

```
cogworks/
├── README.md           # Installation, basic usage
├── DESIGN.md           # Architecture choices
└── examples/
    └── quickstart.py   # Runnable example

mettagrid/
├── README.md           # Environment overview, installation
├── BUILDING.md         # C++ build instructions
└── docs/
    └── curriculum.md   # How to create tasks
```

### Root Documentation

```
docs/
├── getting-started.md  # Overall project intro
├── guides/
│   ├── training.md     # Using cogworks
│   ├── environments.md # Using mettagrid
│   └── distributed.md  # Multi-GPU setup
├── api/
│   ├── cogworks.md     # API reference
│   └── mettagrid.md    # Environment API
└── contributing.md     # Development guide
```

## Common Patterns and Best Practices

### Import Organization

```python
# Standard library
import os
from pathlib import Path

# External packages
import numpy as np
import torch

# Our packages (alphabetical)
from cogworks import api
from mettagrid import mettagrid_env
from mettacommon.util import logger

# Local imports
from .utils import helper
```

### Cross-Package Communication

When packages need to interact:

```python
# cogworks/api.py
from typing import Protocol

class Environment(Protocol):
    """Interface that environments must implement."""
    def reset(self): ...
    def step(self, action): ...

# mettagrid/mettagrid_env.py
class MettaGridPufferEnv:  # Implements Environment protocol
    def reset(self): ...
    def step(self, action): ...
```

### Optional Dependencies

Keep packages lightweight with optional features:

```toml
# cogworks/pyproject.toml
[project.optional-dependencies]
wandb = ["wandb>=0.13.0"]
distributed = ["torch-distributed>=2.0"]
all = ["wandb>=0.13.0", "torch-distributed>=2.0"]
```

## Anti-Patterns to Avoid

1. **Deep nesting**: `from cogworks.rl.algorithms.ppo.utils import thing` ❌
2. **Circular imports**: cogworks → mettagrid → cogworks ❌
3. **Kitchen sink common**: Putting everything in mettacommon ❌
4. **Mixed concerns**: Training code in mettagrid ❌
5. **Hidden dependencies**: Web UI importing from cogworks ❌

## Migration Checklist

- [ ] Flatten directory structure
- [ ] Update all imports
- [ ] Create package-specific pyproject.toml files
- [ ] Move tests to package directories
- [ ] Update CI/CD configuration
- [ ] Test independent package installation
- [ ] Update documentation
- [ ] Communicate changes to team

This flattened structure prioritizes developer experience and maintainability while keeping our options open for future growth.
