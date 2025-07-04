# Metta Repository Organization Plan

## Overview

This document outlines the organization structure for the Metta monorepo, balancing maintainability with modularity. We follow PEP 420 namespace packages to avoid `__init__.py` files and maintain a clean import structure.

## Core Principles

1. **Minimize Package Count**: Limit subpackages to logical, cohesive units that could theoretically be deployed independently
2. **Follow PEP 420**: Use namespace packages to avoid `__init__.py` proliferation
3. **Clear Dependencies**: Maintain a directed acyclic graph (DAG) of dependencies
4. **Consistent Structure**: All packages follow the same `src/` layout pattern
5. **Pragmatic Grouping**: Group related functionality to avoid excessive fragmentation

## Package Structure

### Core Packages (5-7 total)

```
metta/
├── common/                 # Shared utilities and base classes
│   └── src/
│       └── metta/
│           └── common/
├── mettagrid/             # C++/PyBind simulation engine
│   ├── src/
│   │   └── metta/
│   │       └── mettagrid/
│   └── CMakeLists.txt
├── ml/                    # Machine learning components
│   └── src/
│       └── metta/
│           └── ml/
│               ├── rl/         # Reinforcement learning
│               ├── trainer/    # Training infrastructure
│               ├── sweep/      # Hyperparameter sweeping
│               └── eval/       # Evaluation tools
├── observatory/           # Web-based visualization frontend
│   ├── src/               # TypeScript/React
│   └── package.json
├── observatory_backend/   # Backend API for Observatory
│   └── src/
│       └── metta/
│           └── observatory_backend/
├── studio/                # Local development UI
│   ├── src/               # TypeScript/React
│   └── package.json
└── studio_backend/        # Backend services for Studio
    └── src/
        └── metta/
            └── studio_backend/
```

### Import Examples

```python
# Clean namespace imports
from metta.common import utils
from metta.ml.rl import PPOTrainer
from metta.ml.sweep import SweepManager
from metta.mettagrid import GridEnvironment
```

## Package Descriptions

### `metta.common`
**Purpose**: Shared utilities, configurations, and base classes used across multiple packages
- Config management and Hydra resolvers
- Logging utilities
- Common data structures
- Base interfaces/protocols

### `metta.mettagrid`
**Purpose**: Core C++ simulation engine with Python bindings
- High-performance grid-based environments
- C++ implementation with PyBind11 interface
- Requires scikit-build-core for compilation

### `metta.ml`
**Purpose**: All machine learning related components
- **rl/**: Reinforcement learning algorithms and wrappers
- **trainer/**: Training loops and infrastructure
- **sweep/**: Hyperparameter optimization and experiment management
- **eval/**: Evaluation metrics and analysis tools

*Note: Grouping ML components reduces package count while maintaining logical boundaries*

### Frontend/Backend Pairs

#### `observatory` + `observatory_backend`
- Production web interface for experiment tracking and visualization
- Dockerized backend service
- Public-facing deployment

#### `studio` + `studio_backend`
- Local development environment
- Rich debugging and analysis tools
- Direct file system access for local workflows

## Dependency Rules

1. `metta.common` has no internal dependencies
2. `metta.mettagrid` depends only on `metta.common`
3. `metta.ml` depends on `metta.common` and `metta.mettagrid`
4. Backend services can depend on any metta.* package
5. Frontend packages are independent TypeScript/Node projects

## Migration Path

1. **Phase 1**: Move `metta/metta/*` → `ml/src/metta/ml/*`
2. **Phase 2**: Consolidate shared utilities into `common/`
3. **Phase 3**: Ensure all packages use `src/` layout
4. **Phase 4**: Remove all `__init__.py` files except where required by tooling

## Rationale

This structure strikes a balance between:
- **Monolith advocates**: Only 5-7 Python packages to maintain
- **Micropackage advocates**: Clear separation of concerns and potential for independent deployment
- **PEP 420 compliance**: Clean namespace packages without `metta/metta` nesting
- **Practical needs**: Frontend/backend separation for deployment while keeping related code together

## Future Considerations

- If a component grows significantly (>10k LOC), consider splitting
- If deployment requirements diverge, packages can be separated
- Monitor dependency complexity and refactor if cycles emerge

## Development Workflow

All packages remain in the monorepo with:
- Shared tooling configuration (pyproject.toml at root for common settings)
- Unified testing and CI/CD
- Consistent code formatting and linting
- Single `uv.lock` for dependency management

This organization provides structure without excessive fragmentation, maintaining the benefits of both monolithic and modular approaches.
