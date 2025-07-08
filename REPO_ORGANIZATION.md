# Metta Repository Organization Plan

## Overview

This document outlines the organization structure for the Metta monorepo, balancing maintainability with modularity.

### Goals of the Structure

Our package structure is designed to achieve these key goals:

#### Technical Goals

1. **Separate Installation**: Enable subpackages to be installed independently (e.g., `pip install metta-common` without
   installing the entire repository)
2. **Consistent Import Syntax**: Maintain clean imports across all packages using the
   `from metta.subpackage import module` pattern
3. **Shared Namespace**: Allow separate packages to contribute to the same `metta.*` namespace while being independently
   distributable

#### Organizational Goals

4. **Mettagrid Export**: Enable straightforward publishing of subpackages for external use. For example, `mettagrid` can
   be published to PyPI while maintaining the same import syntax (`from metta.mettagrid import ...`) for external users
5. **Independent Development**: Each subpackage has its own `pyproject.toml` and can be developed, tested, and versioned
   independently while sharing the monorepo's tooling
6. **Clear Boundaries**: Enforce separation between the RL system and web services, preventing accidental coupling while
   allowing intentional shared code

While we don't currently publish these packages to PyPI, this structure positions us to do so in the future. For now, we
use `uv sync` with workspace pyproject files for internal development.

### Future Considerations

This organization is a living proposal that we can adapt as our needs evolve:

- The root namespace may eventually migrate from `metta` to `softmax` to align with our organization
- The structure can be adjusted based on actual usage patterns and pain points we discover
- The goal is to align the team "for now" while maintaining flexibility for future changes

This proposal doesn't need to be perfect—it needs to help us work effectively today while keeping our options open for
tomorrow.

## Core Principles

1. **Minimize Package Count**: Limit subpackages to logical, cohesive units that could theoretically be deployed
   independently
2. **Minimize Hierarchy**: Keep namespace as flat as possible for developer ergonomics
   - **10+ subfolder rule**: Only add an extra layer of hierarchy when a directory has more than 10 subfolders
3. **Clear Dependencies**: Maintain a directed acyclic graph (DAG) of dependencies
4. **Consistent Structure**: All packages follow the same `src/` layout pattern (per uv best practices)
5. **Pragmatic Grouping**: Group related functionality to avoid excessive fragmentation
6. **Pragmatic PEP 420**: Follow PEP 420 patterns in subpackages but use `__init__.py` files as needed

## Development Workflow

All packages remain in the monorepo with:

- Shared tooling configuration
- Unified testing and CI/CD
- Consistent code formatting and linting
- Single `uv.lock` for dependency management
- Common development environment setup

## Package Structure

```
metta/
├── src/                  # Core Python package
├── common/               # Shared utilities
├── mettagrid/            # C++ environment
├── backend/              # Backend services
├── apps/                 # All applications (web, desktop, etc.)
│   ├── shared/           # Shared app code
│   ├── observatory/      # Production web app
│   ├── mettascope/       # Replay viewer
│   └── studio/           # Development UI
├── configs/              # Hydra configurations
├── tools/                # CLI tools
├── recipes/              # Example programs
├── docs/                 # Documentation
└── devops/               # Deployment/setup
```

### Package Details

```
metta/
├── src/                   # Main package code
│   ├── __init__.py        # Configures imports to expose metta.* namespace
│   ├── api.py             # API definitions
│   ├── rl/                # Reinforcement learning
│   ├── sweep/             # Hyperparameter sweeping
│   ├── setup/             # Setup tools
│   ├── agent/             # Policy code
│   ├── map/               # Map tools
│   └── eval/              # Evaluation tools
├── tests/                 # Tests for metta/src
├── configs/               # Hydra config DI hierarchy
├── tools/                 # CLI tools (hydra interfaces for running commands)
├── recipes/               # Example programs (complete training/eval scripts)
├── docs/                  # Documentation
├── devops/                # Machine and cloud setup
├── common/                # Shared utilities package (PEP420)
│   └── src/
│       └── metta/
│           └── common/    # metta.common namespace
├── mettagrid/             # C++/PyBind environment package (PEP420)
│   ├── src/
│   │   └── metta/
│   │       └── mettagrid/ # metta.mettagrid namespace
│   └── CMakeLists.txt
├── backend/
│   ├── src/
│   │   └── metta/
│   │       └── backend/
│   │           ├── observatory/   # Observatory API endpoints
│   │           ├── sweep_names/   # Name registration service
│   │           └── stat_buffer/   # Data persistence layer
│   └── docker/
│       └── observatory/
│           ├── Dockerfile
│           └── requirements.txt   # Cherry-pick only needed deps
├── apps/                  # All user-facing applications
│   ├── shared/            # Shared components and utilities
│   │   ├── components/    # Reusable React components
│   │   ├── utils/         # Common TypeScript utilities
│   │   ├── styles/        # Shared CSS/styling
│   │   └── hooks/         # Shared React hooks
│   ├── observatory/       # Production monitoring web app
│   │   ├── src/           # TypeScript/React source
│   │   ├── server.py      # Python server (if needed)
│   │   └── package.json
│   ├── mettascope/        # Web replay visualization
│   │   ├── src/           # TypeScript source
│   │   ├── server.py      # Local Python server
│   │   ├── replays.py     # Replay handling
│   │   └── package.json
│   └── studio/            # Local development UI
│       ├── src/           # TypeScript source
│       ├── server.py      # Local Python server
│       └── package.json
└── pyproject.toml         # Root configuration
```

## Testing Strategy

- Each package maintains its own tests/ directory
- Shared test utilities live in metta.common.testing
- Integration tests live in the root tests/ directory

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

- **Location**: Root-level `src/` directory
- **Namespace**: `metta.*`
- **Purpose**: Core ML functionality

### Shared Utilities (`metta.common`)

- **Location**: `common/` directory
- **Namespace**: `metta.common`
- **Purpose**: Shared utilities that all packages can depend on

  - Config management and Hydra resolvers
  - Logging utilities
  - Base interfaces and protocols
  - Data structures used across packages

_Note: Separate package allows mettagrid to depend on common utils without circular dependency_

### What Goes in `metta.common`?

Only truly shared utilities belong here:

- Logging configuration used by multiple packages
- Common type definitions and protocols
- Shared configuration management
- Utilities needed by both mettagrid and training code

**Examples of what belongs:**

- `logger.py` - Logging configuration used by multiple packages
- `types.py` - Type definitions like `AgentID`, `Position`
- `config.py` - Shared configuration management
- `metrics.py` - Common metric definitions

**Examples of what doesn't belong:**

- `trainer.py` - Training logic (goes in main `metta`)
- `api_utils.py` - Web service utilities (goes in `backend`)
- `react_helpers.js` - Frontend utilities (goes in `apps/shared`)

**Important**: If code is only used by web services, it belongs in `apps/shared/`, not `metta.common`.

### Environment Engine (`metta.mettagrid`)

- **Location**: `mettagrid/` directory
- **Namespace**: `metta.mettagrid`
- **Purpose**: Core C++ simulation engine with Python bindings

  - High-performance grid-based environments
  - C++ implementation with PyBind11 interface
  - Depends on `metta.common` but not main `metta` package

### Backend Services (`metta.backend`)

- **Location**: `backend/` directory
- **Namespace**: `metta.backend`
- **Purpose**: Unified backend services for all server-side functionality

#### Deployment Strategy

Services can be deployed independently using optional dependencies:

```python
# backend/pyproject.toml
[project.optional-dependencies]
observatory = ["fastapi", "uvicorn", "pydantic"]
sweep-names = ["redis", "msgpack"]
stat-buffer = ["sqlalchemy", "psycopg2"]
all = ["fastapi", "uvicorn", "pydantic", "redis", "msgpack", "sqlalchemy", "psycopg2"]
```

## Frontend Applications

Each app includes:

- `src/`: TypeScript/React application code
- `server.py`: Local development server (not part of core packages)
- `package.json`: Node dependencies

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

## Version Management

- All Python packages share version from root pyproject.toml
- Frontend apps version independently via package.json
- Releases coordinate all package versions together

## Dependency Rules

```mermaid
graph TD
    A[metta] --> B[metta.mettagrid]
    A --> C[metta.common]
    B --> C
    D[metta.backend] --> C
    E[Frontend Apps] --> C
```

1. `metta.common` has no internal dependencies (base utilities)
2. `metta.mettagrid` depends only on `metta.common`
3. Main `metta` package depends on `metta.common` and `metta.mettagrid`
4. `metta.backend` depends only on `metta.common`
5. Frontend packages are independent TypeScript/Node projects
6. Frontend `server.py` files can import from any metta.\* package

If `metta.backend` needs functionality from the main package, move the required code to `metta.common`.

### System Boundaries

Our architecture maintains clear boundaries between major systems:

- **RL System**: `metta` (training) + `mettagrid` (environment) + `common` (shared utils)
- **Web Services**: `apps/*` + `backend/*` + `common` (shared utils)
- **Key Rule**: Web services should never import from the main `metta` package directly

This separation ensures that:

- Mettagrid can be exported without pulling in training code
- Web services remain decoupled from RL implementation details
- Shared code is explicitly identified and minimal

### External Library Dependencies

Remember that python has no built-in tree shaking for dependencies! To keep packages lightweight and minimize dependency
bloat:

```python
# ❌ AVOID: Importing entire packages
import torch
import numpy as np
from sklearn import *

# ✅ PREFERRED: Import only specific submodules/functions
from torch.nn import functional as F
from numpy import array, zeros
from sklearn.metrics import accuracy_score
```

**Package-specific rules:**

- `metta.common`: Import only stdlib and minimal utilities
- `metta.backend`: Import only specific web framework components
  ```python
  # Good: from fastapi import FastAPI, HTTPException
  # Bad:  import fastapi
  ```
- `metta.mettagrid`: Import only essential numerical operations
  ```python
  # Good: from numpy import ndarray, zeros
  # Bad:  import numpy
  ```

This approach:

- Reduces memory usage (only loads needed submodules)
- Speeds up import times
- Makes dependencies explicit and easier to audit
- Helps identify if a package is getting too heavy

## PEP 420 Strategy

### Package Structure

- **Main package** (`src/`): Traditional package with `__init__.py`
- **Subpackages** (`common/`, `mettagrid/`, `backend/`): Follow PEP 420 patterns with `__init__.py` as needed

### Directory Structure Rationale

The main `metta` package uses a simplified structure:

```
metta/
├── src/
│   ├── __init__.py        # Traditional package (not PEP 420)
│   ├── api.py             # Imports as: from metta import api
│   └── rl/
└── pyproject.toml         # name = "metta"
```

Subpackages require the "nested" structure to properly install into the `metta` namespace:

```
common/
├── src/
│   └── metta/             # Required: establishes namespace
│       └── common/        # Required: creates metta.common
│           └── utils.py   # Imports as: from metta.common import utils
└── pyproject.toml         # name = "metta.common"
```

**Why the difference?**

- The main package _owns_ the `metta` namespace via its `__init__.py` and package name
- Subpackages must _extend_ into the existing namespace, requiring the full path
- This hybrid approach avoids unnecessary nesting in the main package while maintaining proper namespace organization
  for separately-installable subpackages

**Installation result:**

```
site-packages/
└── metta/
    ├── __init__.py        # From main package
    ├── api.py             # From main package
    ├── common/            # From metta.common package
    │   └── utils.py
    └── mettagrid/         # From metta.mettagrid package
```

**Common mistake:** If subpackages used a flat structure like:

```
common/
├── src/
│   ├── __init__.py
│   └── utils.py
└── pyproject.toml         # name = "metta.common"
```

This would install as `metta/common.py` (a module) rather than `metta/common/` (a package), breaking the intended
namespace structure.

### Implementation Philosophy

1. Structure packages following PEP 420 patterns
2. Use `__init__.py` when helpful for tooling compatibility
3. Focus on clean architecture over strict compliance
4. Remove `__init__.py` files as tooling improves

### Exporting Subpackages

When we need to publish a subpackage (e.g., mettagrid) for external users:

1. The subpackage already has its own `pyproject.toml` with `name = "metta-mettagrid"`
2. We can publish directly to PyPI from the monorepo: `cd mettagrid && uv publish`
3. External users install via `pip install metta-mettagrid`
4. Import syntax remains identical: `from metta.mettagrid import MettaGridPufferEnv`

This approach maintains consistency between internal monorepo development and external usage, while avoiding the
maintenance burden of separate repositories.

## Documentation Strategy

### Documentation Types and Purposes

We maintain three complementary documentation systems:

1. **README.md files**: User-facing documentation for quick understanding and usage
2. **CLAUDE.md files**: AI assistant context for architectural decisions and complex patterns
3. **docs/ folder**: Comprehensive guides, references, and detailed documentation

### Package-Level Documentation

Each package maintains its own documentation suited to its complexity:

```
package/
├── README.md              # Required: Package overview and usage
├── CLAUDE.md              # Optional
└── src/
    └── complex_module/
        └── README.md      # Optional
```

#### README.md Guidelines

- **Required** for all packages
- Focus on: Installation, basic usage, API examples
- Keep concise (under 500 lines)
- Link to `docs/` for detailed information
- Include package-specific badges and status

#### CLAUDE.md Guidelines

- **Optional** - only create when package has:
  - Complex architectural decisions
  - Non-obvious design patterns
  - Domain-specific knowledge (e.g., C++ bindings, RL algorithms)
  - Common implementation pitfalls
- Focus on: Design rationale, patterns, gotchas
- Update when architecture changes or patterns evolve

### Root Documentation Folder

The `docs/` folder serves as our comprehensive documentation hub:

```
docs/
├── README.md              # Documentation index and navigation
├── api/                   # API references and examples
│   ├── reference.md       # Auto-generated API docs
│   └── examples.md        # Curated usage examples
├── guides/                # Step-by-step user guides
│   ├── quickstart.md
│   ├── installation.md
│   └── mapgen.md          # Feature-specific guides
├── metrics/               # Metrics and monitoring documentation
│   ├── README.md          # Metrics overview
│   └── wandb/             # Auto-generated WandB metric docs
├── development/           # Developer and contributor docs
│   ├── architecture.md    # System design documentation
│   ├── contributing.md
│   └── workflows/        # Development workflows
└── assets/               # Images, diagrams, and media
```

### Documentation Hierarchy

Follow this decision tree for where to document:

```mermaid
flowchart TD
    Start{Is it about using<br/>the package?}
    Start -->|YES| PkgReadme[Package README.md]
    Start -->|NO| CompGuide{Is it a<br/>comprehensive guide?}

    PkgReadme --> Complex{Complex topic?}
    Complex -->|YES| LinkDocs[Link to docs/guides/]

    CompGuide -->|YES| DocsGuides[docs/guides/]
    CompGuide -->|NO| ApiRef{Is it API<br/>reference?}

    ApiRef -->|YES| DocsApi[docs/api/]
    ApiRef -->|NO| DocsDev[docs/development/]
```

### Maintenance Protocol

1. **When adding features**: Update package README.md
2. **When changing architecture**: Update relevant CLAUDE.md
3. **When adding complex workflows**: Create guide in docs/
4. **When metrics change**: Auto-regenerate docs/metrics/
5. **During major refactors**: Review all three documentation levels

### Anti-Patterns to Avoid

- ❌ Duplicating content between README and docs/
- ❌ Creating CLAUDE.md for simple utility packages
- ❌ Putting user guides in CLAUDE.md
- ❌ Nesting documentation more than necessary
- ❌ Mixing auto-generated and manual content in same file

This documentation strategy ensures users can quickly understand any package while maintaining detailed references for
complex topics and providing AI assistants with the context they need to help effectively.
