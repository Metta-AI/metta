# Metta Repository Organization Plan

## Overview

This document outlines the organization structure for the Metta monorepo, balancing maintainability with modularity.

## Core Principles

1. **Minimize Package Count**: Limit subpackages to logical, cohesive units that could theoretically be deployed independently
2. **Minimize Hierarchy**: Keep namespace as flat as possible for developer ergonomics
   - **10+ subfolder rule**: Only add an extra layer of hierarchy when a directory has more than 10 subfolders
3. **Clear Dependencies**: Maintain a directed acyclic graph (DAG) of dependencies
4. **Consistent Structure**: All packages follow the same `src/` layout pattern (per uv best practices)
5. **Pragmatic Grouping**: Group related functionality to avoid excessive fragmentation
6. **Pragmatic PEP 420**: Follow PEP 420 patterns in subpackages but use `__init__.py` files as needed

## Package Structure

### Core Packages (6 total)

```
metta/
├── src/                   # Main package code
│   ├── __init__.py        # Configures imports to expose metta.* namespace
│   ├── api.py             # API definitions
│   ├── rl/                # Reinforcement learning
│   ├── sweep/             # Hyperparameter sweeping
│   ├── setup/             # Setup tools
│   ├── map/               # Map tools
│   └── eval/              # Evaluation tools
├── tests/                 # Tests for metta/src
├── configs/               # Hydra config DI hierarchy
├── tools/                 # Hydra CLI interface for API
├── recipes/               # Common training and evaluation programs
├── docs/                  # Documentation
├── devops/                # Machine and cloud setup
├── common/                # Shared utilities package
│   └── src/
│       └── metta/
│           └── common/    # metta.common namespace
├── mettagrid/             # C++/PyBind environment package
│   ├── src/
│   │   └── metta/
│   │       └── mettagrid/ # metta.mettagrid namespace
│   └── CMakeLists.txt
├── backend/               # Remote services package
│   └── src/
│       └── metta/
│           └── backend/   # metta.backend namespace
│               ├── observatory/  # Observatory API endpoints
│               ├── sweep-names/  # Name registration service
│               └── stat-buffer/  # Data persistence layer
├── observatory/           # Web visualization
│   ├── src/               # TypeScript/React
│   └── package.json
├── mettascope/            # Web gameplay visualization
│   ├── src/               # TypeScript source
│   ├── server.py          # Local Python server
│   ├── replays.py         # Replay handling
│   ├── index.html
│   ├── package.json
│   └── tsconfig.json
├── studio/                # Local development UI
│   ├── src/               # TypeScript source
│   ├── server.py          # Local Python server
│   ├── index.html
│   ├── package.json
│   └── tsconfig.json
└── pyproject.toml         # Root configuration
```

### Import Examples

```python
# All imports use metta.* namespace
from metta.common import utils
from metta.rl import MettaTrainer
from metta.sweep import SweepManager
from metta.mettagrid import MettaGridPufferEnv
from metta.backend.sweep_names import SweepNameRegistry
from metta.backend.observatory import RemoteStatsDb
```

## Package Descriptions

### Main Package (`metta`)
**Location**: Root-level `src/` directory
**Namespace**: `metta.*`
**Purpose**: Core ML functionality

- **rl/**: Reinforcement learning algorithms and wrappers
- **sweep/**: Hyperparameter optimization and experiment management
- **setup/**: Environment and dependency setup tools
- **map/**: Map generation and manipulation tools
- **eval/**: Evaluation metrics and analysis tools
- **api.py**: Core API definitions and interfaces

### Shared Utilities (`metta.common`)
**Location**: `common/` directory
**Namespace**: `metta.common`
**Purpose**: Shared utilities that all packages can depend on

- Config management and Hydra resolvers
- Logging utilities
- Base interfaces and protocols
- Data structures used across packages

*Note: Separate package allows mettagrid to depend on common utils without circular dependency*

### Environment Engine (`metta.mettagrid`)
**Location**: `mettagrid/` directory
**Namespace**: `metta.mettagrid`
**Purpose**: Core C++ simulation engine with Python bindings

- High-performance grid-based environments
- C++ implementation with PyBind11 interface
- Depends on `metta.common` but not main `metta` package

### Backend Services (`metta.backend`)
**Location**: `backend/` directory
**Namespace**: `metta.backend`
**Purpose**: Unified backend services for all server-side functionality

- **observatory/**: API endpoints for experiment tracking and visualization
- **sweep-names/**: Process name registration service for sweeps
- **stat-buffer/**: Data persistence and database interfaces

### Frontend Applications

#### Observatory
- Production web interface for experiment tracking and visualization
- Connects to deployed backend service
- Public-facing deployment

#### Mettascope
- Local visualization and replay tool
- Real-time agent observation and replay analysis
- Direct file system access for replay files

#### Studio
- Next-generation local development UI
- Map creation tools
- Enhanced debugging and analysis capabilities

## Dependency Rules

```mermaid
graph TD
    A[metta]->B[metta.mettagrid]
    A -> C[metta.common]
    B --> C
    D[metta.backend] --> B
    E[Frontend Apps] --> B
```

1. `metta.common` has no internal dependencies (base utilities)
2. `metta.mettagrid` depends only on `metta.common`
3. Main `metta` package depends on `metta.common` and `metta.mettagrid`
4. `metta.backend` can import from all metta.* packages
5. Frontend packages are independent TypeScript/Node projects
6. Frontend `server.py` files can import from any metta.* package

## PEP 420 Strategy

### Package Structure
- **Main package** (`src/`): Traditional package with `__init__.py`
- **Subpackages** (`common/`, `mettagrid/`, `backend/`): Follow PEP 420 patterns with `__init__.py` as needed

### Implementation Philosophy
1. Structure packages following PEP 420 patterns
2. Use `__init__.py` when helpful for tooling compatibility
3. Focus on clean architecture over strict compliance
4. Remove `__init__.py` files as tooling improves


## Development Workflow

All packages remain in the monorepo with:
- Shared tooling configuration
- Unified testing and CI/CD
- Consistent code formatting and linting
- Single `uv.lock` for dependency management
- Common development environment setup

## Future Considerations

- **Scaling**: If a component exceeds 10k LOC, consider splitting
- **Deployment**: If requirements diverge, packages can be separated
- **Dependencies**: Monitor complexity and refactor if cycles emerge
- **Standards**: Adopt full PEP 420 compliance as tooling matures

## Benefits

This structure provides:
- **Clean imports**: Simple, predictable namespace
- **Minimal nesting**: Easy navigation and discovery
- **Clear boundaries**: Well-defined package responsibilities
- **Flexibility**: Can evolve without major restructuring
- **Pragmatism**: Works with current tooling while preparing for future
