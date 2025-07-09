# Metta Repository Organization Plan

## Overview

A flattened monorepo structure with enforced `metta.*` namespace imports and consistent `metta-*` PyPI package naming.

### Key Principles

1. **Enforced Namespace**: All imports MUST use `metta.` prefix (enforced by linting)
2. **Flat Structure**: Top-level package directories without src/ nesting
3. **Consistent Naming**: PyPI packages use `metta-` prefix
4. **PEP 420 Namespaces**: Symlinks enable the `metta.*` import pattern

## Repository Structure

```
metta/
├── cogworks/                   # RL training framework
├── mettagrid/                  # C++/Python environment
├── common/                     # Shared utilities
├── backend-shared/             # Shared backend services
├── gridworks/                  # Map editor
├── observatory/                # Production monitoring
├── mettascope/                 # Replay viewer
├── metta/                      # PEP 420 namespace (symlinks only)
│   ├── cogworks → ../cogworks
│   ├── mettagrid → ../mettagrid
│   ├── common → ../common
│   └── ...                     # Other symlinks
├── tools/                      # Standalone scripts
├── recipes/                    # Example scripts and workflows
├── configs/                    # Hydra configurations
├── scenes/                     # Map/scene definitions
├── docs/                       # Documentation
├── devops/                     # Infrastructure
└── pyproject.toml              # Workspace configuration
```

## Detailed Structure

```
metta/
├── cogworks/                   # RL training framework (merged from metta/ + agent/)
│   ├── agent/                  # From agent/src/metta/agent/
│   ├── rl/                     # From metta/rl/
│   ├── eval/                   # From metta/eval/
│   ├── sweep/                  # From metta/sweep/
│   ├── sim/                    # From metta/sim/
│   ├── mapgen/                 # From metta/map/
│   ├── tests/                  # Combined tests
│   ├── pyproject.toml          # name = "metta-cogworks"
│   └── api.py                  # Main API (from metta/api.py)
│
├── mettagrid/                  # C++/Python environment
│   ├── tests/
│   ├── benchmarks/
│   ├── configs/
│   ├── *.py                    # Flattened Python files
│   └── pyproject.toml          # name = "metta-mettagrid"
│
├── common/                     # Shared Python utilities
│   ├── util/                   # From common/src/metta/common/util/
│   ├── profiling/              # From common/src/metta/common/profiling/
│   ├── wandb/                  # From common/src/metta/common/wandb/
│   ├── tests/
│   └── pyproject.toml          # name = "metta-common"
│
├── backend-shared/             # Shared backend services
│   ├── sweep_names.py          # Name registration service
│   ├── stat_buffer.py          # Data persistence layer
│   ├── auth.py                 # Authentication utilities
│   ├── database.py             # Database connection pooling
│   ├── cache.py                # Caching utilities
│   ├── utils.py                # General backend utilities
│   ├── tests/
│   └── pyproject.toml          # name = "metta-backend-shared"
│
├── ui-shared/                  # Shared UI components for web apps
│   ├── components/             # Reusable React components
│   ├── hooks/                  # Shared React hooks
│   ├── utils/                  # Common TypeScript utilities
│   ├── styles/                 # Shared CSS/styling
│   └── package.json            # Shared UI dependencies
│
├── gridworks/                  # Map editor (from studio/)
│   ├── src/                    # TypeScript/React frontend
│   ├── public/
│   ├── server.py               # Python server
│   ├── pyproject.toml          # name = "metta-gridworks" (new)
│   └── package.json
│
├── observatory/                # Production monitoring
│   ├── src/                    # React frontend
│   ├── api/                    # Observatory-specific backend
│   │   ├── endpoints.py        # Observatory API endpoints
│   │   └── requirements.txt    # API-specific Python dependencies
│   ├── pyproject.toml          # name = "metta-observatory"
│   ├── package.json
│   └── Dockerfile              # From backend/docker/observatory/
│
├── mettascope/                 # Replay viewer
│   ├── src/                    # TypeScript source
│   ├── data/                   # Assets
│   ├── tools/                  # Python tools
│   ├── server.py               # Python replay server
│   ├── replays.py              # Replay handling
│   └── package.json
│
├── metta/                      # PEP 420 namespace package (NO __init__.py)
│   ├── cogworks → ../cogworks         # Symlink to ../cogworks
│   ├── mettagrid → ../mettagrid       # Symlink to ../mettagrid
│   ├── common → ../common              # Symlink to ../common
│   ├── backend_shared → ../backend-shared  # Symlink to ../backend-shared
│   ├── gridworks → ../gridworks       # Symlink to ../gridworks
│   ├── observatory → ../observatory   # Symlink to ../observatory
│   └── mettascope → ../mettascope     # Symlink to ../mettascope
│
├── tools/                      # Standalone scripts (train.py, sweep_*.py, etc.)
├── recipes/                    # Example scripts and workflows
├── configs/                    # Hydra configurations
├── scenes/                     # Map generation patterns
├── docs/                       # Documentation
├── devops/                     # Infrastructure
├── setup/                      # From metta/setup/
├── pyproject.toml              # Workspace configuration
└── README.md                   # Mono-repo overview
```

## Import Convention (ENFORCED)

All imports MUST use the `metta.` namespace prefix:

```python
# ✅ CORRECT (enforced by linting)
from metta.cogworks import api
from metta.cogworks.rl import trainer
from metta.cogworks.agent import MettaAgent
from metta.common.util import config
from metta.mettagrid import MettaGridPufferEnv
from metta.backend_shared import sweep_names

# ❌ INCORRECT (blocked by linting)
from cogworks import api  # Will fail lint check
import mettagrid  # Will fail lint check
from common import logger  # Will fail lint check
```

A custom lint rule will enforce this convention across the entire codebase.

## Package Examples

### Main Training Framework

```toml
# cogworks/pyproject.toml
[project]
name = "metta-cogworks"
version = "0.1.0"
description = "Metta RL training framework"
```

### Common Utilities

```toml
# common/pyproject.toml
[project]
name = "metta-common"
version = "0.1.0"
description = "Shared utilities for Metta packages"
```

### Backend Shared Services

```toml
# backend-shared/pyproject.toml
[project]
name = "metta-backend-shared"
version = "0.1.0"
description = "Shared backend services for Metta applications"

[tool.setuptools]
packages = ["backend_shared"]
```

## Current → New Mapping

```
# CURRENT LOCATION                      → NEW LOCATION
metta/src/api.py                       → cogworks/api.py
metta/src/rl/                          → cogworks/rl/
metta/src/sweep/                       → cogworks/sweep/
metta/src/setup/                       → cogworks/setup/
metta/src/agent/                       → cogworks/agent/
metta/src/map/                         → cogworks/mapgen/
metta/src/eval/                        → cogworks/eval/
metta/tests/                           → cogworks/tests/
metta/configs/                         → configs/
metta/tools/                           → tools/
metta/recipes/                         → recipes/
metta/docs/                            → docs/
metta/devops/                          → devops/

common/src/metta/common/               → common/
mettagrid/src/metta/mettagrid/         → mettagrid/

backend/src/metta/backend/sweep_names/ → backend-shared/sweep_names.py
backend/src/metta/backend/stat_buffer/ → backend-shared/stat_buffer.py
backend/src/metta/backend/observatory/ → observatory/api/endpoints.py
backend/docker/observatory/            → observatory/Dockerfile

apps/shared/                           → ui-shared/
apps/observatory/                      → observatory/
apps/mettascope/                       → mettascope/
apps/studio/                           → gridworks/

# NAMESPACE SYMLINKS (NEW)
(none)                                 → metta/cogworks → ../cogworks
(none)                                 → metta/mettagrid → ../mettagrid
(none)                                 → metta/common → ../common
(none)                                 → metta/backend_shared → ../backend-shared
```

## Installation

```bash
# Individual packages
pip install metta-cogworks
pip install metta-mettagrid
pip install metta-common
pip install metta-backend-shared

# Development setup
uv sync  # Installs all workspace packages
```

## PEP 420 Namespace Setup

The `metta/` directory is a PEP 420 implicit namespace package:
- NO `__init__.py` file
- Contains only symlinks to actual packages
- Enables `metta.*` imports without code duplication

**Creating the symlinks:**
```bash
mkdir -p metta
cd metta
ln -s ../cogworks cogworks
ln -s ../mettagrid mettagrid
ln -s ../common common
ln -s ../backend-shared backend_shared
ln -s ../gridworks gridworks
ln -s ../observatory observatory
ln -s ../mettascope mettascope
```

## Key Benefits

1. **Consistent Namespace**: All code uses `metta.*` imports
2. **Brand Recognition**: Clear `metta-` prefix for all packages
3. **Flat Structure**: Simple directory layout
4. **No Ambiguity**: Enforced import style prevents confusion
5. **Easy Migration**: Clear path from current structure

## Migration Path

1. Create `metta/` namespace directory with symlinks
2. Update all imports to use `metta.` prefix
3. Add lint rule to enforce `metta.` imports
4. Update `pyproject.toml` files with `metta-` names
5. Move `recipes/` to root level
6. Create `backend-shared/` package
7. Test with `uv sync`
8. Update CI/CD and documentation

This structure provides a clean, professional organization with enforced consistency across the entire codebase.
