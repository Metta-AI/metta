# Metta Repository Organization Plan (Revised)

## Overview

This document outlines the organization structure for the Metta monorepo, balancing maintainability with modularity. We follow PEP 420 namespace packages to avoid `__init__.py` files and maintain a clean import structure.

## Core Principles

1. **Minimize Package Count**: Limit subpackages to logical, cohesive units that could theoretically be deployed independently
2. **Minimize Hierarchy**: Keep namespace as flat as possible for developer ergonomics
3. **Clear Dependencies**: Maintain a directed acyclic graph (DAG) of dependencies
4. **Consistent Structure**: All packages follow the same `src/` layout pattern (per uv best practices)
5. **Pragmatic Grouping**: Group related functionality to avoid excessive fragmentation

*Note: Strict PEP 420 compliance is not a requirement. We can implement the structure with temporary `__init__.py` files until tooling support is complete*

## Package Structure

### Core Packages (7 total)

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
├── backend/               # Unified backend services
│   └── src/
│       └── metta/
│           └── backend/
│               ├── observatory/  # Observatory API endpoints
│               ├── naming/       # Name registration service
│               ├── storage/      # Data persistence layer
│               └── api/          # Common API infrastructure
├── observatory/           # Web-based visualization frontend
│   ├── src/               # TypeScript/React
│   └── package.json
├── mettascope/            # Existing local visualization tool
│   ├── src/               # TypeScript source
│   ├── server.py          # Local Python server
│   ├── replays.py         # Replay handling
│   ├── index.html
│   ├── package.json
│   └── tsconfig.json
└── studio/                # Enhanced development UI
    ├── src/               # TypeScript source
    ├── server.py          # Local Python server
    ├── index.html
    ├── package.json
    └── tsconfig.json
```

### Import Examples

```python
# Clean namespace imports
from metta.common import utils
from metta.ml.rl import PPOTrainer
from metta.ml.sweep import SweepManager
from metta.mettagrid import GridEnvironment
from metta.backend.naming import NameRegistry
from metta.backend.observatory import ExperimentTracker
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

### `metta.backend`
**Purpose**: Unified backend services for all server-side functionality
- **observatory/**: API endpoints for experiment tracking and visualization
- **naming/**: Process name registration service for sweeps
- **storage/**: Shared data persistence and database interfaces
- **api/**: Common API infrastructure (authentication, routing, middleware)

*Benefits of unified backend:*
- Single deployment unit for all services
- Shared infrastructure (database connections, auth, etc.)
- Easier service-to-service communication
- Consistent API patterns

### Frontend Packages

#### `observatory`
- Production web interface for experiment tracking and visualization
- Connects to deployed backend service
- Public-facing deployment

#### `mettascope`
- Existing local visualization and replay tool
- Focused on real-time agent observation and replay analysis
- Direct file system access for replay files
- Lightweight, proven architecture

#### `studio`
- Next-generation development environment
- Enhanced debugging and analysis capabilities
- Broader scope than Mettascope (full experiment lifecycle)
- May eventually supersede Mettascope but both coexist for now

## Dependency Rules

1. `metta.common` has no internal dependencies
2. `metta.mettagrid` depends only on `metta.common`
3. `metta.ml` depends on `metta.common` and `metta.mettagrid`
4. `metta.backend` can depend on any metta.* package
5. Frontend packages are independent TypeScript/Node projects
6. `studio/server.py` and `mettascope/server.py` can import from any metta.* package for local serving

## Migration Path

1. **Phase 1**: Remove fast_gae dependency (no longer used)
2. **Phase 2**: Move `metta/metta/*` → `ml/src/metta/ml/*`
3. **Phase 3**: Consolidate shared utilities into `common/`
4. **Phase 4**: Create unified `backend/` package from existing services
5. **Phase 5**: Keep Mettascope as-is, develop Studio in parallel
6. **Phase 6**: Keep minimal `__init__.py` files until scikit-build-core PR #808 lands
7. **Phase 7**: Remove all `__init__.py` files once tooling support is available

## PEP 420 Strategy

### Current State
- **scikit-build-core limitation**: Currently prevents full PEP 420 compliance in editable installs
- **Active development**: PR #808 shows this is being actively addressed
- **fast_gae removal**: Eliminates C++ compilation needs in the main package

### Implementation Approach
1. **Structure for PEP 420**: Organize packages following PEP 420 patterns
2. **Temporary workaround**: Keep `__init__.py` files only where absolutely necessary
3. **Future cleanup**: Remove `__init__.py` files once scikit-build-core support lands

### Benefits of This Approach
- Standards-compliant structure from day one
- Minimal technical debt
- Easy migration path (just delete `__init__.py` files later)
- Better positioning for PyPI publishing if needed in future

## Rationale

This structure strikes a balance between:
- **Standards compliance**: PEP 420-ready structure positions us for the future
- **Pragmatism**: Temporary `__init__.py` files work with current tooling
- **Developer experience**: Minimal hierarchy within each package
- **Operational simplicity**: Unified backend, proven frontend patterns

## Future Considerations

- If a component grows significantly (>10k LOC), consider splitting
- If deployment requirements diverge, packages can be separated
- Monitor dependency complexity and refactor if cycles emerge
- Remove `__init__.py` files once scikit-build-core support lands

## Development Workflow

All packages remain in the monorepo with:
- Shared tooling configuration (pyproject.toml at root for common settings)
- Unified testing and CI/CD
- Consistent code formatting and linting
- Single `uv.lock` for dependency management

This organization provides structure without excessive fragmentation, maintaining the benefits of both monolithic and modular approaches.
