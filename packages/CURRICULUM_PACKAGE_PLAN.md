# Curriculum System Package Migration Plan

## Executive Summary

This document outlines the comprehensive plan to extract the curriculum learning system from `metta/cogworks/curriculum/` into a standalone, reusable package. The curriculum system provides adaptive task generation, learning progress tracking, and curriculum-based training for reinforcement learning agents.

---

## Package Name

**Selected Name: `agora`** ⭐

- **Meaning**: Ancient Greek central public space and marketplace where citizens gathered for assembly, learning, and exchange of ideas
- **Rationale**:
  - Perfectly captures the concept of a centralized learning system
  - Short, memorable, and easy to type
  - Evokes the classical tradition of education and knowledge sharing
  - Unique and available on PyPI
  - Philosophical connection to adaptive learning environments where agents "gather" to learn
- **Package import**: `import agora`

The name `agora` symbolizes a place where learning paths converge, knowledge is exchanged, and adaptive curriculum strategies emerge - much like how the ancient Agora served as the heart of civic and intellectual life.

---

## Current State Analysis

### File Structure
```
metta/cogworks/curriculum/
├── __init__.py              # Public API exports
├── curriculum.py            # Core curriculum logic (523 lines)
├── curriculum_env.py        # PufferEnv wrapper (small)
├── learning_progress_algorithm.py  # LP algorithm (379 lines)
├── lp_scorers.py           # Learning progress scorers
├── shared_memory_backend.py # Multiprocess coordination
├── stats.py                # Statistics and analytics
├── task_generator.py       # Task generation (419 lines)
├── task_tracker.py         # Task tracking logic
├── demo.py                 # Usage examples
└── structure.md            # Documentation (empty)
```

### Key Dependencies

**Internal (Metta-specific)**:
- `mettagrid.config.mettagrid_config.MettaGridConfig`
- `mettagrid.base_config.Config`
- `mettagrid.util.module.load_symbol`

**External (Python ecosystem)**:
- `pufferlib.PufferEnv` (for environment wrapper)
- `pydantic` (configuration management)
- `numpy` (numerical operations)
- Standard library: `abc`, `logging`, `multiprocessing`, `typing`

### Usage Patterns

The curriculum system is used in **47 files** across the codebase:
- Training recipes (`experiments/recipes/`)
- Core RL training (`metta/rl/training/`)
- Simulation systems (`metta/sim/`)
- Tests (`tests/cogworks/curriculum/`)
- Gridworks UI (`metta/gridworks/`)

**Common import patterns**:
```python
from metta.cogworks.curriculum import (
    Curriculum, CurriculumConfig,
    LearningProgressAlgorithm, LearningProgressConfig,
    TaskGenerator, BucketedTaskGenerator,
)
```

---

## Migration Architecture

### Package Structure

Following the pattern from `cogames` and `mettagrid`:

```
packages/agora/
├── pyproject.toml           # Package metadata, dependencies, build config
├── README.md                # Package documentation
├── LICENSE                  # MIT license
├── CHANGELOG.md             # Version history
├── src/
│   └── agora/
│       ├── __init__.py      # Public API
│       ├── py.typed         # Type stub marker
│       ├── curriculum.py    # Core curriculum classes
│       ├── algorithms/
│       │   ├── __init__.py
│       │   ├── base.py      # Abstract algorithm interface
│       │   ├── learning_progress.py
│       │   └── scorers.py   # LP scorers
│       ├── generators/
│       │   ├── __init__.py
│       │   ├── base.py      # TaskGenerator ABC
│       │   ├── single.py    # SingleTaskGenerator
│       │   ├── bucketed.py  # BucketedTaskGenerator
│       │   └── set.py       # TaskGeneratorSet
│       ├── tracking/
│       │   ├── __init__.py
│       │   ├── tracker.py   # TaskTracker
│       │   ├── stats.py     # Statistics collection
│       │   └── memory.py    # Shared memory backend
│       ├── wrappers/
│       │   ├── __init__.py
│       │   └── puffer.py    # PufferEnv wrapper
│       └── config.py        # Configuration protocols
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_curriculum_core.py
│   ├── test_algorithms.py
│   ├── test_generators.py
│   ├── test_tracking.py
│   ├── test_checkpointing.py
│   └── test_integration.py
├── examples/
│   ├── basic_curriculum.py
│   ├── learning_progress.py
│   └── custom_generator.py
└── docs/
    ├── getting_started.md
    ├── api_reference.md
    ├── algorithms.md
    └── examples.md
```

---

## Design Decisions

### 1. Environment Configuration Independence

**Problem**: Current code tightly couples to `MettaGridConfig`

**Solution**: Use Protocol/ABC for environment configuration
```python
from typing import Protocol, TypeVar, Generic

TConfig = TypeVar('TConfig')

class TaskConfig(Protocol):
    """Protocol for task configurations."""
    def model_copy(self, *, deep: bool = False) -> 'TaskConfig': ...
    def model_dump(self) -> dict: ...

class Curriculum(Generic[TConfig]):
    """Generic curriculum that works with any config type."""
    def __init__(self, task_generator: TaskGenerator[TConfig]) -> None:
        ...
```

**Backward compatibility**: Provide mettagrid-specific helpers in separate submodule:
```python
# agora.mettagrid module
from mettagrid.config import MettaGridConfig
from agora import TaskGenerator

def mettagrid_curriculum(config: MettaGridConfig) -> Curriculum[MettaGridConfig]:
    """Create curriculum for MettaGrid environments."""
    ...
```

### 2. Dependency Management

**Core dependencies** (minimal):
- `numpy` - numerical operations
- `pydantic>=2.0` - configuration validation
- Python 3.11+ (match metta requirements)

**Optional dependencies**:
```toml
[project.optional-dependencies]
puffer = ["pufferlib-core"]  # For PufferEnv wrapper
mettagrid = ["mettagrid"]    # For mettagrid integration
dev = ["pytest", "pytest-xdist", "ruff", "mypy"]
```

### 3. Public API Design

**Minimal, clean API** following Python best practices:

```python
# Core exports
from agora import (
    # Main classes
    Curriculum,
    CurriculumConfig,
    CurriculumTask,

    # Algorithms
    CurriculumAlgorithm,
    LearningProgressAlgorithm,
    LearningProgressConfig,

    # Task generation
    TaskGenerator,
    SingleTaskGenerator,
    BucketedTaskGenerator,
    TaskGeneratorSet,

    # Tracking
    TaskTracker,
    StatsLogger,
    SliceAnalyzer,

    # Wrappers (optional)
    CurriculumEnv,  # Requires pufferlib-core
)
```

### 4. Versioning Strategy

Use `setuptools_scm` (following existing package pattern):
- Version tags: `agora-v0.1.0`, `agora-v0.2.0`, etc.
- Auto-generate version from git tags
- Semantic versioning (MAJOR.MINOR.PATCH)

Initial release: `v0.1.0` (beta/development status)

---

## Migration Steps

### Phase 1: Package Setup (Week 1)

#### Step 1.1: Create Package Structure
```bash
cd packages/
mkdir -p agora/{src/agora,tests,examples,docs}
```

#### Step 1.2: Create `pyproject.toml`
Based on `cogames/pyproject.toml` template:
```toml
[build-system]
requires = ["setuptools>=61.0", "wheel", "setuptools_scm>=8"]
build-backend = "setuptools.build_meta"

[project]
name = "agora"
description = "Adaptive curriculum learning for reinforcement learning agents"
readme = "README.md"
requires-python = ">=3.11,<3.13"
dependencies = [
    "numpy>=2.0.0",
    "pydantic>=2.11.5",
]
dynamic = ["version"]

[project.optional-dependencies]
puffer = ["pufferlib-core"]
mettagrid = ["mettagrid"]
dev = [
    "pytest>=8.3.3",
    "pytest-xdist>=3.8.0",
    "ruff>=0.7.0",
    "mypy>=1.11.0",
]

[tool.setuptools.packages.find]
where = ["src"]
include = ["agora", "agora.*"]

[tool.setuptools.package-data]
agora = ["py.typed"]

[tool.setuptools_scm]
tag_regex = "^agora-v(?P<version>\\d+\\.\\d+\\.\\d+(?:\\.\\d+)?)$"
version_scheme = "no-guess-dev"
local_scheme = "node-and-date"
root = "../.."
fallback_version = "0.0.0"

[tool.setuptools_scm.scm.git]
describe_command = [
    "git", "describe", "--dirty", "--tags", "--long",
    "--match", "agora-v*",
]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]

[tool.uv.sources]
mettagrid = { workspace = true, optional = true }
```

#### Step 1.3: Create Initial Files
- `README.md` - Package overview, quickstart
- `LICENSE` - MIT license (copy from mettagrid)
- `src/agora/py.typed` - Empty marker file
- `.gitignore` - Standard Python ignores

### Phase 2: Code Migration (Week 1-2)

#### Step 2.1: Refactor Configuration Abstraction

Create `src/agora/config.py`:
```python
from typing import Protocol, TypeVar, runtime_checkable

@runtime_checkable
class TaskConfig(Protocol):
    """Protocol defining task configuration interface."""
    def model_copy(self, *, deep: bool = False) -> 'TaskConfig': ...
    def model_dump(self, **kwargs) -> dict: ...
    @classmethod
    def model_validate(cls, obj: dict) -> 'TaskConfig': ...

TConfig = TypeVar('TConfig', bound=TaskConfig)
```

#### Step 2.2: Migrate Core Files

Copy and refactor in order:
1. `task_tracker.py` → `src/agora/tracking/tracker.py`
   - No mettagrid dependencies
   - Update imports

2. `shared_memory_backend.py` → `src/agora/tracking/memory.py`
   - No external dependencies
   - Clean up

3. `stats.py` → `src/agora/tracking/stats.py`
   - Generic implementation
   - Update imports

4. `lp_scorers.py` → `src/agora/algorithms/scorers.py`
   - Update imports to use `TaskTracker` from new location

5. `learning_progress_algorithm.py` → `src/agora/algorithms/learning_progress.py`
   - Refactor to use generic config types
   - Update imports

6. `task_generator.py` → Split into:
   - `src/agora/generators/base.py` (ABC)
   - `src/agora/generators/single.py`
   - `src/agora/generators/bucketed.py`
   - `src/agora/generators/set.py`
   - Make generic over config type

7. `curriculum.py` → `src/agora/curriculum.py`
   - Make generic: `Curriculum[TConfig]`
   - Update all imports

8. `curriculum_env.py` → `src/agora/wrappers/puffer.py`
   - Make optional (requires pufferlib)
   - Add import guards

#### Step 2.3: Create Public API

`src/agora/__init__.py`:
```python
"""Adaptive curriculum learning for RL agents."""

__version__ = "0.1.0"  # Auto-generated by setuptools_scm

# Core curriculum
from .curriculum import (
    Curriculum,
    CurriculumConfig,
    CurriculumTask,
    CurriculumAlgorithm,
    CurriculumAlgorithmConfig,
)

# Algorithms
from .algorithms.learning_progress import (
    LearningProgressAlgorithm,
    LearningProgressConfig,
)
from .algorithms.scorers import (
    LPScorer,
    BasicLPScorer,
    BidirectionalLPScorer,
)

# Task generation
from .generators.base import TaskGenerator, TaskGeneratorConfig
from .generators.single import SingleTaskGenerator
from .generators.bucketed import BucketedTaskGenerator, Span
from .generators.set import TaskGeneratorSet

# Tracking
from .tracking.tracker import TaskTracker
from .tracking.stats import StatsLogger, SliceAnalyzer

# Configuration
from .config import TaskConfig

# Wrappers (optional imports)
try:
    from .wrappers.puffer import CurriculumEnv
    __all__ += ["CurriculumEnv"]
except ImportError:
    pass

__all__ = [
    # Core
    "Curriculum",
    "CurriculumConfig",
    "CurriculumTask",
    "CurriculumAlgorithm",
    "CurriculumAlgorithmConfig",
    # Algorithms
    "LearningProgressAlgorithm",
    "LearningProgressConfig",
    "LPScorer",
    "BasicLPScorer",
    "BidirectionalLPScorer",
    # Generators
    "TaskGenerator",
    "TaskGeneratorConfig",
    "SingleTaskGenerator",
    "BucketedTaskGenerator",
    "TaskGeneratorSet",
    "Span",
    # Tracking
    "TaskTracker",
    "StatsLogger",
    "SliceAnalyzer",
    # Config
    "TaskConfig",
]
```

### Phase 3: Testing Migration (Week 2)

#### Step 3.1: Migrate Test Suite

Copy tests from `tests/cogworks/curriculum/` to `packages/agora/tests/`:
- `test_curriculum_core.py`
- `test_curriculum_algorithms.py`
- `test_curriculum_checkpointing.py`
- `test_curriculum_env.py`
- `test_curriculum_invariants.py`
- `test_lp_config_overrides.py`
- `test_serialization.py`
- `test_helpers.py`
- `conftest.py`

#### Step 3.2: Update Test Imports

Replace all:
```python
# Old
from metta.cogworks.curriculum import Curriculum

# New
from agora import Curriculum
```

#### Step 3.3: Add mettagrid Integration Tests

Create `tests/test_mettagrid_integration.py`:
```python
"""Test integration with mettagrid (requires optional dependency)."""
pytest.importorskip("mettagrid")

from mettagrid.config import MettaGridConfig
from agora import Curriculum, SingleTaskGenerator
...
```

#### Step 3.4: Run Test Suite

```bash
cd packages/agora
uv run pytest tests/
```

### Phase 4: Documentation (Week 2)

#### Step 4.1: Create README.md

Include:
- Project overview
- Key features
- Installation instructions
- Quick start example
- Links to detailed docs

#### Step 4.2: Create docs/

- `getting_started.md` - Tutorial for new users
- `api_reference.md` - API documentation
- `algorithms.md` - Learning progress algorithm details
- `examples.md` - Usage examples and recipes

#### Step 4.3: Create examples/

Copy and adapt from `demo.py`:
- `basic_curriculum.py` - Simple single-task example
- `learning_progress.py` - LP algorithm example
- `custom_generator.py` - Custom task generator
- `mettagrid_example.py` - MettaGrid integration

### Phase 5: Integration with Metta Workspace (Week 3)

#### Step 5.1: Add to Workspace

Update root `pyproject.toml` to include agora in workspace:
```toml
[tool.uv.workspace]
members = [
    "packages/mettagrid",
    "packages/pufferlib-core",
    "packages/cogames",
    "packages/agora",  # NEW
    # ... other packages
]
```

#### Step 5.2: Update Metta Dependencies

Update metta's `pyproject.toml`:
```toml
dependencies = [
    "agora",  # NEW
    # ... other deps
]

[tool.uv.sources]
agora = { workspace = true }
```

#### Step 5.3: Create Compatibility Shim

Keep `metta/cogworks/curriculum/` as a thin wrapper for backward compatibility:

`metta/cogworks/curriculum/__init__.py`:
```python
"""
DEPRECATED: Import from 'agora' package instead.

This module provides backward compatibility. It will be removed in future versions.
"""
import warnings

warnings.warn(
    "Importing from metta.cogworks.curriculum is deprecated. "
    "Use 'import agora' instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from agora
from agora import *  # noqa: F403, F401

__all__ = [
    "Curriculum",
    "CurriculumConfig",
    # ... all exports
]
```

#### Step 5.4: Update Usage in Metta Codebase

Gradually update imports across codebase:
```python
# Old
from metta.cogworks.curriculum import Curriculum

# New
from agora import Curriculum
```

Files to update (47 total):
- `metta/rl/training/training_environment.py`
- `metta/rl/vecenv.py`
- `metta/sim/simulation.py`
- All recipe files in `experiments/recipes/`
- All test files

**Strategy**: Create migration script:
```bash
# migration_script.sh
find . -type f -name "*.py" -exec sed -i '' \
  's/from metta\.cogworks\.curriculum/from agora/g' {} \;
```

### Phase 6: Testing & Validation (Week 3)

#### Step 6.1: Run All Tests
```bash
# Test agora package
cd packages/agora
uv run pytest tests/ -v

# Test metta integration
cd ../..
uv run pytest tests/cogworks/curriculum/ -v

# Run full test suite
uv run pytest tests/ -v
```

#### Step 6.2: Test Training Recipes
```bash
# Quick end-to-end test
timeout 30s uv run ./tools/run.py experiments.recipes.arena.train run=test

# Test curriculum-specific recipes
timeout 30s uv run ./tools/run.py experiments.recipes.cogs_v_clips.level_1 run=test
```

#### Step 6.3: Lint and Format
```bash
cd packages/agora
ruff format src/ tests/
ruff check --fix src/ tests/
mypy src/
```

### Phase 7: Publishing Preparation (Week 4)

#### Step 7.1: Version Tag
```bash
git tag -a agora-v0.1.0 -m "Initial release of agora package"
git push origin agora-v0.1.0
```

#### Step 7.2: Build Package
```bash
cd packages/agora
uv build
# Creates dist/agora-0.1.0.tar.gz and dist/agora-0.1.0-py3-none-any.whl
```

#### Step 7.3: Test Installation
```bash
# Test in isolated environment
uv venv test-env
source test-env/bin/activate
uv pip install dist/agora-0.1.0-py3-none-any.whl
python -c "import agora; print(agora.__version__)"
```

#### Step 7.4: PyPI Publishing (Optional)

If making public:
```bash
# Test PyPI first
uv publish --repository testpypi dist/*

# Production PyPI
uv publish dist/*
```

---

## Backward Compatibility Strategy

### Deprecation Timeline

**v0.1.0 - Initial release**:
- New `agora` package available
- Old `metta.cogworks.curriculum` kept with deprecation warnings
- All internal metta code updated to use `agora`

**v0.2.0 - Deprecation period** (3-6 months):
- Deprecation warnings continue
- External users migrate their code
- Documentation highlights migration path

**v1.0.0 - Removal** (6-12 months):
- Remove `metta.cogworks.curriculum` module
- Only `agora` package remains

### Migration Guide for Users

Create `docs/migration_guide.md`:

```markdown
# Migration Guide: metta.cogworks.curriculum → agora

## Quick Migration

### Before (deprecated)
```python
from metta.cogworks.curriculum import (
    Curriculum,
    CurriculumConfig,
    LearningProgressAlgorithm,
)
```

### After
```python
from agora import (
    Curriculum,
    CurriculumConfig,
    LearningProgressAlgorithm,
)
```

## API Changes

Most APIs remain identical. Key changes:

1. **Generic configs**: Curriculum now generic over config type
2. **Optional dependencies**: PufferEnv wrapper requires `pip install agora[puffer]`
3. **MettaGrid helpers**: Use `from agora.mettagrid import ...` (requires `pip install agora[mettagrid]`)
```

---

## Detailed Implementation Plan

This section provides file-by-file implementation details with specific code changes needed for each component.

### File Migration Matrix

| Source File | Target File | Dependencies | Complexity | Est. Time |
|-------------|-------------|--------------|------------|-----------|
| `task_tracker.py` | `src/agora/tracking/tracker.py` | None | Low | 1h |
| `shared_memory_backend.py` | `src/agora/tracking/memory.py` | None | Low | 1h |
| `stats.py` | `src/agora/tracking/stats.py` | `tracker.py` | Medium | 2h |
| `lp_scorers.py` | `src/agora/algorithms/scorers.py` | `tracker.py` | Low | 1h |
| `learning_progress_algorithm.py` | `src/agora/algorithms/learning_progress.py` | `scorers.py`, `stats.py` | High | 4h |
| `task_generator.py` | `src/agora/generators/*.py` | None | High | 4h |
| `curriculum.py` | `src/agora/curriculum.py` | All above | High | 4h |
| `curriculum_env.py` | `src/agora/wrappers/puffer.py` | `curriculum.py` | Low | 1h |

**Total estimated implementation time: 18-20 hours**

---

### Implementation Order & Details

#### 1. Create Package Infrastructure (Day 1, Morning)

**Files to create:**

##### `packages/agora/pyproject.toml`
```toml
[build-system]
requires = ["setuptools>=61.0", "wheel", "setuptools_scm>=8"]
build-backend = "setuptools.build_meta"

[project]
name = "agora"
description = "Adaptive curriculum learning for reinforcement learning agents"
readme = "README.md"
requires-python = ">=3.11,<3.13"
authors = [
    { name = "Metta AI Team", email = "team@metta.ai" }
]
license = { text = "MIT" }
keywords = [
    "reinforcement-learning",
    "curriculum-learning",
    "adaptive-learning",
    "machine-learning",
]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
dependencies = [
    "numpy>=2.0.0",
    "pydantic>=2.11.5",
]
dynamic = ["version"]

[project.optional-dependencies]
puffer = ["pufferlib-core"]
mettagrid = ["mettagrid"]
all = ["pufferlib-core", "mettagrid"]
dev = [
    "pytest>=8.3.3",
    "pytest-xdist>=3.8.0",
    "pytest-cov>=5.0.0",
    "ruff>=0.7.0",
    "mypy>=1.11.0",
]

[project.urls]
Homepage = "https://github.com/Metta-AI/metta"
Documentation = "https://github.com/Metta-AI/metta/tree/main/packages/agora"
Repository = "https://github.com/Metta-AI/metta"
Issues = "https://github.com/Metta-AI/metta/issues"

[tool.setuptools.packages.find]
where = ["src"]
include = ["agora", "agora.*"]

[tool.setuptools.package-data]
agora = ["py.typed"]

[tool.setuptools_scm]
tag_regex = "^agora-v(?P<version>\\d+\\.\\d+\\.\\d+(?:\\.\\d+)?)$"
version_scheme = "no-guess-dev"
local_scheme = "node-and-date"
root = "../.."
fallback_version = "0.1.0"

[tool.setuptools_scm.scm.git]
describe_command = [
    "git", "describe", "--dirty", "--tags", "--long",
    "--match", "agora-v*",
]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_functions = ["test_*"]
python_classes = ["Test*"]
addopts = "--strict-markers --cov=agora --cov-report=term-missing"

[tool.ruff]
line-length = 120
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP"]
ignore = ["E501"]

[tool.mypy]
python_version = "3.11"
strict = true
warn_return_any = true
warn_unused_configs = true

[tool.uv.sources]
mettagrid = { workspace = true, optional = true }
pufferlib-core = { workspace = true, optional = true }
```

**Checklist:**
- [ ] Create `packages/agora/pyproject.toml`
- [ ] Create `packages/agora/README.md`
- [ ] Create `packages/agora/LICENSE` (copy from mettagrid)
- [ ] Create `packages/agora/src/agora/py.typed` (empty file)
- [ ] Create `packages/agora/.gitignore`

---

#### 2. Configuration Abstraction Layer (Day 1, Afternoon)

**File: `packages/agora/src/agora/config.py`**

```python
"""Configuration protocols for environment-agnostic curriculum system."""

from typing import Any, Protocol, TypeVar, runtime_checkable

__all__ = ["TaskConfig", "TConfig"]


@runtime_checkable
class TaskConfig(Protocol):
    """
    Protocol defining the interface for task configurations.

    Any configuration class used with agora must implement these methods.
    Pydantic BaseModel classes automatically satisfy this protocol.
    """

    def model_copy(self, *, deep: bool = False) -> "TaskConfig":
        """Create a copy of this configuration."""
        ...

    def model_dump(
        self,
        *,
        mode: str = "python",
        include: Any = None,
        exclude: Any = None,
        by_alias: bool = False,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
        round_trip: bool = False,
        warnings: bool = True,
    ) -> dict[str, Any]:
        """Serialize configuration to dictionary."""
        ...

    @classmethod
    def model_validate(cls, obj: Any) -> "TaskConfig":
        """Validate and create configuration from dictionary."""
        ...


# Generic type variable bound to TaskConfig protocol
TConfig = TypeVar("TConfig", bound=TaskConfig)
```

**Changes from original:**
- New file, doesn't exist in current codebase
- Provides abstraction over `MettaGridConfig`
- Uses Protocol for duck typing

**Checklist:**
- [ ] Create `packages/agora/src/agora/config.py`
- [ ] Add docstrings
- [ ] Add type hints

---

#### 3. Task Tracker (Day 1-2)

**Source:** `metta/cogworks/curriculum/task_tracker.py` (458 lines)
**Target:** `packages/agora/src/agora/tracking/tracker.py`

**Key changes:**

```python
# OLD IMPORTS (lines 1-11)
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from metta.cogworks.curriculum.shared_memory_backend import LocalMemoryBackend, SharedMemoryBackend, TaskMemoryBackend

# NEW IMPORTS
from __future__ import annotations

import logging
import time
from typing import Any

from agora.tracking.memory import LocalMemoryBackend, SharedMemoryBackend, TaskMemoryBackend
```

**Detailed changes:**
1. Update all imports to use `agora.tracking.memory` instead of `metta.cogworks.curriculum.shared_memory_backend`
2. Replace `Dict, List, Optional` with lowercase `dict, list` (modern Python typing)
3. No other logic changes needed

**Files to modify:**
- Copy `task_tracker.py` → `tracking/tracker.py`
- Update imports
- Format with ruff

**Checklist:**
- [ ] Create `packages/agora/src/agora/tracking/__init__.py`
- [ ] Copy `task_tracker.py` to `packages/agora/src/agora/tracking/tracker.py`
- [ ] Update import: `shared_memory_backend` → `agora.tracking.memory`
- [ ] Update type hints to use lowercase (dict, list, etc.)
- [ ] Run `ruff format` and `ruff check --fix`
- [ ] Verify no mettagrid dependencies

---

#### 4. Shared Memory Backend (Day 2)

**Source:** `metta/cogworks/curriculum/shared_memory_backend.py` (283 lines)
**Target:** `packages/agora/src/agora/tracking/memory.py`

**Key changes:**

```python
# OLD IMPORTS (lines 1-11)
from __future__ import annotations

import hashlib
import logging
import struct
from abc import ABC, abstractmethod
from multiprocessing import RLock, shared_memory
from typing import Any, ContextManager, Optional

import numpy as np

# NEW IMPORTS (no changes needed - all stdlib/numpy)
from __future__ import annotations

import hashlib
import logging
import struct
from abc import ABC, abstractmethod
from multiprocessing import RLock, shared_memory
from typing import Any, ContextManager

import numpy as np
import numpy.typing as npt
```

**Detailed changes:**
1. Add `numpy.typing` for better type hints
2. Replace `Optional[X]` with `X | None`
3. No other changes needed - this file has no internal dependencies

**Checklist:**
- [ ] Copy `shared_memory_backend.py` to `packages/agora/src/agora/tracking/memory.py`
- [ ] Update type hints to use modern syntax (`X | None`)
- [ ] Add `numpy.typing` import
- [ ] Run `ruff format` and `ruff check --fix`

---

#### 5. Statistics Collection (Day 2)

**Source:** `metta/cogworks/curriculum/stats.py` (469 lines)
**Target:** `packages/agora/src/agora/tracking/stats.py`

**Key changes:**

```python
# OLD IMPORTS (lines 1-12)
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

if TYPE_CHECKING:
    from metta.cogworks.curriculum.task_tracker import TaskTracker

# NEW IMPORTS
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from agora.tracking.tracker import TaskTracker
```

**Detailed changes:**
1. Update TYPE_CHECKING import: `metta.cogworks.curriculum.task_tracker` → `agora.tracking.tracker`
2. Update all type hints to use lowercase (dict, list)
3. Remove unused `Optional, Dict, List` imports
4. No logic changes

**Checklist:**
- [ ] Copy `stats.py` to `packages/agora/src/agora/tracking/stats.py`
- [ ] Update TYPE_CHECKING import path
- [ ] Update type hints
- [ ] Run `ruff format` and `ruff check --fix`
- [ ] Update `tracking/__init__.py` to export `StatsLogger`, `SliceAnalyzer`

---

#### 6. Learning Progress Scorers (Day 2)

**Source:** `metta/cogworks/curriculum/lp_scorers.py`
**Target:** `packages/agora/src/agora/algorithms/scorers.py`

**Key changes:**

```python
# OLD IMPORTS (lines 1-15)
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

from .task_tracker import TaskTracker

if TYPE_CHECKING:
    pass

# NEW IMPORTS
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from agora.tracking.tracker import TaskTracker

if TYPE_CHECKING:
    pass
```

**Detailed changes:**
1. Update import: `from .task_tracker` → `from agora.tracking.tracker`
2. Update type hints
3. No logic changes

**Checklist:**
- [ ] Create `packages/agora/src/agora/algorithms/__init__.py`
- [ ] Copy `lp_scorers.py` to `packages/agora/src/agora/algorithms/scorers.py`
- [ ] Update TaskTracker import path
- [ ] Update type hints
- [ ] Run `ruff format` and `ruff check --fix`

---

#### 7. Task Generators (Day 3)

**Source:** `metta/cogworks/curriculum/task_generator.py` (419 lines)
**Target:** Split into multiple files:
- `packages/agora/src/agora/generators/base.py`
- `packages/agora/src/agora/generators/single.py`
- `packages/agora/src/agora/generators/bucketed.py`
- `packages/agora/src/agora/generators/set.py`

This is the most complex refactoring. Here's the detailed breakdown:

##### 7a. Base Generator (`generators/base.py`)

Extract base classes and protocols:

```python
"""Base classes and protocols for task generation."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Generic, TypeVar

from pydantic import ConfigDict, Field

from agora.config import TaskConfig, TConfig

__all__ = [
    "TaskGenerator",
    "TaskGeneratorConfig",
    "Span",
]

logger = logging.getLogger(__name__)


class Span:
    """Represents a range [min, max] for parameter variation."""

    def __init__(self, min: float, max: float):
        self.min = min
        self.max = max

    def __repr__(self) -> str:
        return f"Span({self.min}, {self.max})"


class TaskGeneratorConfig(ABC):
    """Base configuration for task generators."""

    model_config = ConfigDict(arbitrary_types_allowed=True)


class TaskGenerator(ABC, Generic[TConfig]):
    """
    Abstract base class for task generation.

    Type parameter TConfig should be a concrete config type that implements TaskConfig protocol.
    """

    Config: ClassVar[type[TaskGeneratorConfig]]

    @abstractmethod
    def generate_task(self, task_id: str) -> TConfig:
        """Generate a task configuration with the given ID."""
        ...

    @abstractmethod
    def get_all_task_ids(self) -> list[str]:
        """Return all possible task IDs this generator can produce."""
        ...
```

**Checklist for base.py:**
- [ ] Create `packages/agora/src/agora/generators/__init__.py`
- [ ] Create `packages/agora/src/agora/generators/base.py`
- [ ] Extract `Span`, `TaskGeneratorConfig`, `TaskGenerator` from original
- [ ] Make `TaskGenerator` generic over `TConfig`
- [ ] Update imports to use `agora.config`

##### 7b. Single Task Generator (`generators/single.py`)

```python
"""Single-task generator for fixed task configurations."""

from __future__ import annotations

import logging
from typing import ClassVar

from pydantic import Field

from agora.config import TConfig
from agora.generators.base import TaskGenerator, TaskGeneratorConfig

__all__ = ["SingleTaskGenerator"]

logger = logging.getLogger(__name__)


class SingleTaskGenerator(TaskGenerator[TConfig]):
    """
    Generates a single fixed task.

    Useful for non-curriculum training or as a baseline.
    """

    class Config(TaskGeneratorConfig):
        """Configuration for single-task generator."""

        env: TConfig = Field(..., description="Fixed task configuration")

    Config: ClassVar[type[Config]]

    def __init__(self, config: Config[TConfig]) -> None:
        self.config = config
        self.task_id = "task_0"

    def generate_task(self, task_id: str) -> TConfig:
        """Return the fixed task configuration."""
        return self.config.env.model_copy(deep=True)

    def get_all_task_ids(self) -> list[str]:
        """Return single task ID."""
        return [self.task_id]
```

**Changes from original:**
1. Extract `SingleTaskGenerator` class and its `Config` from `task_generator.py`
2. Remove dependency on `MettaGridConfig` - use generic `TConfig`
3. Update imports

**Checklist:**
- [ ] Create `packages/agora/src/agora/generators/single.py`
- [ ] Extract SingleTaskGenerator from original file
- [ ] Make it generic over TConfig
- [ ] Update imports
- [ ] Run `ruff format` and `ruff check --fix`

##### 7c. Bucketed Task Generator (`generators/bucketed.py`)

```python
"""Bucketed task generator for curriculum learning."""

from __future__ import annotations

import logging
import random
from typing import Any, ClassVar

from pydantic import Field

from agora.config import TConfig
from agora.generators.base import Span, TaskGenerator, TaskGeneratorConfig

__all__ = ["BucketedTaskGenerator"]

logger = logging.getLogger(__name__)


class BucketedTaskGenerator(TaskGenerator[TConfig]):
    """
    Generates tasks by sampling parameters from bucketed ranges.

    Useful for curriculum learning where difficulty increases across buckets.
    """

    class Config(TaskGeneratorConfig):
        """Configuration for bucketed task generator."""

        env: TConfig = Field(..., description="Base environment configuration")
        buckets: dict[str, list[Any]] = Field(
            default_factory=dict,
            description="Parameter buckets for each field"
        )
        num_tasks_per_bucket: int = Field(
            default=10,
            description="Number of tasks to generate per bucket"
        )

    Config: ClassVar[type[Config]]

    def __init__(self, config: Config[TConfig]) -> None:
        self.config = config
        self.base_config = config.env
        self.buckets = config.buckets
        self._task_cache: dict[str, TConfig] = {}
        self._generate_all_tasks()

    def _generate_all_tasks(self) -> None:
        """Pre-generate all task configurations."""
        # Implementation details from original task_generator.py
        # ... (copy logic from BucketedTaskGenerator)
        pass

    def generate_task(self, task_id: str) -> TConfig:
        """Return cached task configuration."""
        if task_id not in self._task_cache:
            raise ValueError(f"Unknown task_id: {task_id}")
        return self._task_cache[task_id].model_copy(deep=True)

    def get_all_task_ids(self) -> list[str]:
        """Return all pre-generated task IDs."""
        return list(self._task_cache.keys())

    @classmethod
    def from_mg(cls, mg_config: TConfig) -> Config[TConfig]:
        """
        Create bucketed generator config from a base config.

        This helper method can be overridden for specific config types.
        """
        return cls.Config(env=mg_config)
```

**Changes:**
1. Extract `BucketedTaskGenerator` from original
2. Make generic over TConfig
3. Keep all bucketing logic intact
4. Update imports

**Checklist:**
- [ ] Create `packages/agora/src/agora/generators/bucketed.py`
- [ ] Extract BucketedTaskGenerator from original
- [ ] Copy all bucketing logic
- [ ] Make generic over TConfig
- [ ] Update imports
- [ ] Run `ruff format` and `ruff check --fix`

##### 7d. Task Generator Set (`generators/set.py`)

```python
"""Composite generator that combines multiple task generators."""

from __future__ import annotations

import logging
import random
from typing import Any, ClassVar

from pydantic import Field

from agora.config import TConfig
from agora.generators.base import TaskGenerator, TaskGeneratorConfig

__all__ = ["TaskGeneratorSet"]

logger = logging.getLogger(__name__)


class TaskGeneratorSet(TaskGenerator[TConfig]):
    """
    Combines multiple task generators with weighted sampling.

    Useful for mixing different task types in curriculum.
    """

    class Config(TaskGeneratorConfig):
        """Configuration for task generator set."""

        task_generators: list[TaskGeneratorConfig] = Field(
            ...,
            description="List of task generator configs"
        )
        weights: list[float] = Field(
            ...,
            description="Sampling weights for each generator"
        )

    Config: ClassVar[type[Config]]

    def __init__(self, config: Config[TConfig]) -> None:
        self.config = config
        self.generators: list[TaskGenerator[TConfig]] = []
        self.weights = config.weights
        # Initialize generators from configs
        # ... (copy logic from original)

    def generate_task(self, task_id: str) -> TConfig:
        """Generate task from appropriate generator."""
        # Parse task_id to determine which generator
        # ... (copy logic from original)
        pass

    def get_all_task_ids(self) -> list[str]:
        """Return all task IDs from all generators."""
        all_ids = []
        for i, gen in enumerate(self.generators):
            all_ids.extend([f"gen{i}_{tid}" for tid in gen.get_all_task_ids()])
        return all_ids
```

**Checklist:**
- [ ] Create `packages/agora/src/agora/generators/set.py`
- [ ] Extract TaskGeneratorSet from original
- [ ] Make generic over TConfig
- [ ] Update imports
- [ ] Run `ruff format` and `ruff check --fix`

##### 7e. Update generators/__init__.py

```python
"""Task generation for curriculum learning."""

from agora.generators.base import Span, TaskGenerator, TaskGeneratorConfig
from agora.generators.bucketed import BucketedTaskGenerator
from agora.generators.set import TaskGeneratorSet
from agora.generators.single import SingleTaskGenerator

__all__ = [
    "TaskGenerator",
    "TaskGeneratorConfig",
    "SingleTaskGenerator",
    "BucketedTaskGenerator",
    "TaskGeneratorSet",
    "Span",
]
```

---

#### 8. Learning Progress Algorithm (Day 3-4)

**Source:** `metta/cogworks/curriculum/learning_progress_algorithm.py` (379 lines)
**Target:** `packages/agora/src/agora/algorithms/learning_progress.py`

**Key changes:**

```python
# OLD IMPORTS (lines 1-14)
from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional

from .curriculum import CurriculumAlgorithm, CurriculumAlgorithmConfig, CurriculumTask
from .lp_scorers import BasicLPScorer, BidirectionalLPScorer, LPScorer
from .stats import CacheCoordinator, LPStatsAggregator
from .task_tracker import TaskTracker

# NEW IMPORTS
from __future__ import annotations

import logging
import random
from typing import Any, Generic

from agora.algorithms.scorers import BasicLPScorer, BidirectionalLPScorer, LPScorer
from agora.config import TConfig
from agora.curriculum import CurriculumAlgorithm, CurriculumAlgorithmConfig, CurriculumTask
from agora.tracking.stats import CacheCoordinator, LPStatsAggregator
from agora.tracking.tracker import TaskTracker
```

**Detailed changes:**
1. Update all relative imports to absolute `agora.*` imports
2. Make `LearningProgressAlgorithm` generic over `TConfig`
3. Update type hints to use lowercase
4. No logic changes needed

**Checklist:**
- [ ] Copy `learning_progress_algorithm.py` to `packages/agora/src/agora/algorithms/learning_progress.py`
- [ ] Update all imports to use `agora.*` paths
- [ ] Make generic over TConfig where needed
- [ ] Update type hints
- [ ] Run `ruff format` and `ruff check --fix`
- [ ] Update `algorithms/__init__.py` to export classes

---

#### 9. Core Curriculum (Day 4)

**Source:** `metta/cogworks/curriculum/curriculum.py` (523 lines)
**Target:** `packages/agora/src/agora/curriculum.py`

**Key changes:**

```python
# OLD IMPORTS (lines 1-20)
from __future__ import annotations

import abc
import logging
import random
from abc import ABC
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Union

# ... more imports ...

from pydantic import ConfigDict, Field

from metta.cogworks.curriculum.stats import SliceAnalyzer, StatsLogger
from metta.cogworks.curriculum.task_generator import AnyTaskGeneratorConfig, SingleTaskGenerator
from mettagrid.base_config import Config
from mettagrid.config.mettagrid_config import MettaGridConfig

# ... at end of file:
from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressConfig

# NEW IMPORTS
from __future__ import annotations

import abc
import logging
import random
from abc import ABC
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Union

from pydantic import BaseModel, ConfigDict, Field

from agora.config import TaskConfig, TConfig
from agora.tracking.stats import SliceAnalyzer, StatsLogger

if TYPE_CHECKING:
    from agora.algorithms.learning_progress import LearningProgressConfig
    from agora.generators.base import TaskGeneratorConfig
    from agora.generators.single import SingleTaskGenerator
```

**Major refactoring:**

1. **Remove MettaGridConfig dependency:**
```python
# OLD
class CurriculumTask:
    id: str
    config: MettaGridConfig
    metadata: dict[str, Any]

# NEW
class CurriculumTask(Generic[TConfig]):
    id: str
    config: TConfig  # Generic config type
    metadata: dict[str, Any]
```

2. **Make Curriculum generic:**
```python
# OLD
class Curriculum:
    def __init__(
        self,
        task_generator: TaskGenerator,
        algorithm: CurriculumAlgorithm | None = None,
    ):
        ...

# NEW
class Curriculum(Generic[TConfig]):
    def __init__(
        self,
        task_generator: TaskGenerator[TConfig],
        algorithm: CurriculumAlgorithm[TConfig] | None = None,
    ):
        ...
```

3. **Update Config class:**
```python
# OLD (inherits from mettagrid.base_config.Config)
class CurriculumConfig(Config):
    task_generator: AnyTaskGeneratorConfig = ...

# NEW (inherits from pydantic.BaseModel)
class CurriculumConfig(BaseModel, Generic[TConfig]):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    task_generator: TaskGeneratorConfig = ...
```

**Checklist:**
- [ ] Copy `curriculum.py` to `packages/agora/src/agora/curriculum.py`
- [ ] Update all imports to `agora.*`
- [ ] Make `CurriculumTask` generic over TConfig
- [ ] Make `Curriculum` generic over TConfig
- [ ] Make `CurriculumConfig` generic over TConfig
- [ ] Replace `mettagrid.base_config.Config` with `pydantic.BaseModel`
- [ ] Update all type hints
- [ ] Run `ruff format` and `ruff check --fix`
- [ ] Run `mypy` to check type consistency

---

#### 10. PufferEnv Wrapper (Day 4)

**Source:** `metta/cogworks/curriculum/curriculum_env.py`
**Target:** `packages/agora/src/agora/wrappers/puffer.py`

**Key changes:**

```python
# OLD IMPORTS
from __future__ import annotations

from typing import Any

from pufferlib import PufferEnv

from .curriculum import Curriculum

# NEW IMPORTS
from __future__ import annotations

from typing import Any, Generic

try:
    from pufferlib import PufferEnv
    PUFFERLIB_AVAILABLE = True
except ImportError:
    PUFFERLIB_AVAILABLE = False
    PufferEnv = object  # type: ignore

from agora.config import TConfig
from agora.curriculum import Curriculum


if not PUFFERLIB_AVAILABLE:
    raise ImportError(
        "pufferlib-core is required to use CurriculumEnv. "
        "Install with: pip install agora[puffer]"
    )


class CurriculumEnv(PufferEnv, Generic[TConfig]):
    """
    PufferEnv wrapper that integrates with agora curriculum system.

    Requires: pip install agora[puffer]
    """

    def __init__(
        self,
        curriculum: Curriculum[TConfig],
        **puffer_kwargs: Any
    ):
        self.curriculum = curriculum
        super().__init__(**puffer_kwargs)
        # ... rest of implementation
```

**Checklist:**
- [ ] Create `packages/agora/src/agora/wrappers/__init__.py`
- [ ] Copy `curriculum_env.py` to `packages/agora/src/agora/wrappers/puffer.py`
- [ ] Add optional import guard for pufferlib
- [ ] Make `CurriculumEnv` generic over TConfig
- [ ] Update imports
- [ ] Run `ruff format` and `ruff check --fix`

---

#### 11. Main Package __init__.py (Day 4)

**File:** `packages/agora/src/agora/__init__.py`

```python
"""
Agora: Adaptive Curriculum Learning for Reinforcement Learning

Agora provides a flexible curriculum learning system for RL agents,
featuring adaptive task generation, learning progress tracking, and
seamless integration with popular RL frameworks.

Example:
    >>> from agora import Curriculum, SingleTaskGenerator
    >>> generator = SingleTaskGenerator(config=my_config)
    >>> curriculum = Curriculum(task_generator=generator)
    >>> task = curriculum.sample_task()
"""

from __future__ import annotations

__version__ = "0.1.0"  # Managed by setuptools_scm

# Core curriculum
from agora.curriculum import (
    Curriculum,
    CurriculumAlgorithm,
    CurriculumAlgorithmConfig,
    CurriculumConfig,
    CurriculumTask,
)

# Algorithms
from agora.algorithms.learning_progress import (
    LearningProgressAlgorithm,
    LearningProgressConfig,
)
from agora.algorithms.scorers import (
    BasicLPScorer,
    BidirectionalLPScorer,
    LPScorer,
)

# Task generation
from agora.generators.base import Span, TaskGenerator, TaskGeneratorConfig
from agora.generators.bucketed import BucketedTaskGenerator
from agora.generators.set import TaskGeneratorSet
from agora.generators.single import SingleTaskGenerator

# Tracking
from agora.tracking.stats import SliceAnalyzer, StatsLogger
from agora.tracking.tracker import TaskTracker

# Configuration
from agora.config import TaskConfig

# Optional wrappers
_optional_exports: list[str] = []

try:
    from agora.wrappers.puffer import CurriculumEnv
    _optional_exports.append("CurriculumEnv")
except ImportError:
    pass

__all__ = [
    # Version
    "__version__",
    # Core
    "Curriculum",
    "CurriculumConfig",
    "CurriculumTask",
    "CurriculumAlgorithm",
    "CurriculumAlgorithmConfig",
    # Algorithms
    "LearningProgressAlgorithm",
    "LearningProgressConfig",
    "LPScorer",
    "BasicLPScorer",
    "BidirectionalLPScorer",
    # Generators
    "TaskGenerator",
    "TaskGeneratorConfig",
    "SingleTaskGenerator",
    "BucketedTaskGenerator",
    "TaskGeneratorSet",
    "Span",
    # Tracking
    "TaskTracker",
    "StatsLogger",
    "SliceAnalyzer",
    # Config
    "TaskConfig",
] + _optional_exports
```

**Checklist:**
- [ ] Create `packages/agora/src/agora/__init__.py`
- [ ] Export all public APIs
- [ ] Handle optional imports gracefully
- [ ] Add module docstring with examples
- [ ] Verify all exports work

---

### Metta Integration Changes

#### 12. Update Root Workspace Configuration

**File: `pyproject.toml` (root)**

Add agora to workspace members:

```toml
[tool.uv.workspace]
members = [
    "agent",
    "app_backend",
    "common",
    "experiments",
    "packages/cogames",
    "packages/cortex",
    "packages/mettagrid",
    "packages/pufferlib-core",
    "packages/agora",  # <-- ADD THIS
]
```

**Checklist:**
- [ ] Edit root `pyproject.toml`
- [ ] Add `"packages/agora"` to workspace members
- [ ] Run `uv sync` to verify workspace integration

---

#### 13. Add Agora Dependency to Metta

**File: `pyproject.toml` (root, or metta-specific if exists)**

```toml
[project]
dependencies = [
    # ... existing deps ...
    "agora",
]

[tool.uv.sources]
agora = { workspace = true }
```

**Checklist:**
- [ ] Add `agora` to dependencies
- [ ] Add workspace source for agora
- [ ] Run `uv sync`

---

#### 14. Create Backward Compatibility Shim

**File: `metta/cogworks/curriculum/__init__.py`**

Replace entire file with:

```python
"""
DEPRECATED: Import from 'agora' package instead.

This module provides backward compatibility and will be removed in a future version.

Migration:
    OLD: from metta.cogworks.curriculum import Curriculum
    NEW: from agora import Curriculum
"""

from __future__ import annotations

import warnings

warnings.warn(
    "metta.cogworks.curriculum is deprecated. Use 'agora' package instead.\n"
    "  OLD: from metta.cogworks.curriculum import Curriculum\n"
    "  NEW: from agora import Curriculum\n"
    "Install: pip install agora",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from agora for backward compatibility
from agora import *  # noqa: F403, F401
from agora import __all__  # noqa: F401

# Legacy helper functions for MettaGrid
try:
    from mettagrid.config.mettagrid_config import MettaGridConfig
    from agora import SingleTaskGenerator, BucketedTaskGenerator, TaskGeneratorSet, CurriculumConfig

    def single_task(mg_config: MettaGridConfig) -> SingleTaskGenerator.Config:
        """DEPRECATED: Use agora.SingleTaskGenerator.Config directly."""
        return SingleTaskGenerator.Config(env=mg_config.model_copy(deep=True))

    def bucketed(mg_config: MettaGridConfig) -> BucketedTaskGenerator.Config:
        """DEPRECATED: Use agora.BucketedTaskGenerator.Config.from_mg directly."""
        return BucketedTaskGenerator.Config.from_mg(mg_config.model_copy(deep=True))

    def merge(task_generator_configs: list) -> TaskGeneratorSet.Config:
        """DEPRECATED: Use agora.TaskGeneratorSet.Config directly."""
        return TaskGeneratorSet.Config(
            task_generators=task_generator_configs,
            weights=[1.0] * len(task_generator_configs)
        )

    def env_curriculum(mg_config: MettaGridConfig) -> CurriculumConfig:
        """DEPRECATED: Use agora.CurriculumConfig directly."""
        return CurriculumConfig(task_generator=SingleTaskGenerator.Config(env=mg_config))

except ImportError:
    # If mettagrid not available, skip helper functions
    pass
```

**Checklist:**
- [ ] Replace `metta/cogworks/curriculum/__init__.py` with shim
- [ ] Add deprecation warning
- [ ] Re-export all agora APIs
- [ ] Include legacy helper functions
- [ ] Test import still works: `from metta.cogworks.curriculum import Curriculum`

---

#### 15. Delete Old Curriculum Files

**Files to delete:**
- `metta/cogworks/curriculum/curriculum.py`
- `metta/cogworks/curriculum/learning_progress_algorithm.py`
- `metta/cogworks/curriculum/task_generator.py`
- `metta/cogworks/curriculum/task_tracker.py`
- `metta/cogworks/curriculum/stats.py`
- `metta/cogworks/curriculum/lp_scorers.py`
- `metta/cogworks/curriculum/shared_memory_backend.py`
- `metta/cogworks/curriculum/curriculum_env.py`
- `metta/cogworks/curriculum/demo.py`
- `metta/cogworks/curriculum/structure.md`

**Checklist:**
- [ ] Verify backward compatibility shim works
- [ ] Verify all tests pass with shim
- [ ] Delete old implementation files (keep only __init__.py shim)
- [ ] Commit deletion with message: "refactor: migrate curriculum to agora package"

---

#### 16. Update Import Statements in Metta Codebase

**Files to update (47 total):**

Create migration script:

```bash
#!/bin/bash
# migrate_imports.sh

# Find and replace imports
find metta/ tests/ experiments/ -type f -name "*.py" -exec sed -i '' \
  -e 's/from metta\.cogworks\.curriculum import/from agora import/g' \
  -e 's/from metta\.cogworks\.curriculum\./from agora./g' \
  -e 's/import metta\.cogworks\.curriculum/import agora/g' \
  {} \;

echo "Import migration complete. Please review changes with git diff."
```

**Manual review required for these files:**

1. **`metta/rl/training/training_environment.py`**
   ```python
   # OLD
   from metta.cogworks.curriculum import Curriculum, CurriculumConfig

   # NEW
   from agora import Curriculum, CurriculumConfig
   ```

2. **`metta/rl/training/evaluator.py`**
   ```python
   # OLD
   from metta.cogworks.curriculum import Curriculum

   # NEW
   from agora import Curriculum
   ```

3. **`metta/rl/vecenv.py`**
   ```python
   # OLD
   from metta.cogworks.curriculum import CurriculumEnv

   # NEW
   from agora import CurriculumEnv  # Requires agora[puffer]
   ```

4. **`metta/sim/simulation.py`**
   ```python
   # OLD
   from metta.cogworks.curriculum import Curriculum, CurriculumConfig

   # NEW
   from agora import Curriculum, CurriculumConfig
   ```

5. **All recipe files in `experiments/recipes/`:**
   - `experiments/recipes/arena.py`
   - `experiments/recipes/cvc_arena.py`
   - `experiments/recipes/cogs_v_clips/level_1.py`
   - etc. (see grep results for full list)

**Checklist:**
- [ ] Run migration script
- [ ] Review all changes with `git diff`
- [ ] Manually verify each critical file
- [ ] Run linter: `ruff check --fix metta/ tests/ experiments/`
- [ ] Run formatter: `ruff format metta/ tests/ experiments/`

---

### Testing Migration

#### 17. Migrate Test Suite

**Tests to copy from `tests/cogworks/curriculum/` to `packages/agora/tests/`:**

| Source Test | Target Test | Changes Needed |
|-------------|-------------|----------------|
| `test_curriculum_core.py` | `test_curriculum_core.py` | Update imports to `agora` |
| `test_curriculum_algorithms.py` | `test_algorithms.py` | Update imports, move to algorithms/ |
| `test_curriculum_checkpointing.py` | `test_checkpointing.py` | Update imports |
| `test_curriculum_env.py` | `test_puffer_wrapper.py` | Update imports, mark as requiring puffer |
| `test_curriculum_invariants.py` | `test_invariants.py` | Update imports |
| `test_lp_config_overrides.py` | `test_lp_overrides.py` | Update imports |
| `test_curriculum_capacity_eviction.py` | `test_capacity.py` | Update imports |
| `test_curriculum_shared_memory.py` | `test_shared_memory.py` | Update imports |
| `test_serialization.py` | `test_serialization.py` | Update imports |
| `conftest.py` | `conftest.py` | Update imports, fixtures |

**Common import changes for all tests:**

```python
# OLD
from metta.cogworks.curriculum import (
    Curriculum,
    CurriculumConfig,
    LearningProgressAlgorithm,
)
from metta.cogworks.curriculum.task_generator import SingleTaskGenerator
from metta.cogworks.curriculum.task_tracker import TaskTracker

# NEW
from agora import (
    Curriculum,
    CurriculumConfig,
    LearningProgressAlgorithm,
    SingleTaskGenerator,
    TaskTracker,
)
```

**Checklist:**
- [ ] Copy all test files to `packages/agora/tests/`
- [ ] Update all imports
- [ ] Create `conftest.py` with shared fixtures
- [ ] Add pytest markers for optional dependencies:
  ```python
  # In conftest.py
  import pytest

  def pytest_configure(config):
      config.addinivalue_line("markers", "requires_puffer: tests that need pufferlib")
      config.addinivalue_line("markers", "requires_mettagrid: tests that need mettagrid")
  ```
- [ ] Mark tests requiring optional deps:
  ```python
  @pytest.mark.requires_mettagrid
  def test_mettagrid_integration():
      ...
  ```

---

#### 18. Create Integration Tests

**File: `packages/agora/tests/test_mettagrid_integration.py`**

```python
"""Integration tests with MettaGrid (optional dependency)."""

import pytest

mettagrid = pytest.importorskip("mettagrid")

from mettagrid.config.mettagrid_config import MettaGridConfig
from mettagrid.envs import MettaGridEnv

from agora import (
    BucketedTaskGenerator,
    Curriculum,
    CurriculumConfig,
    LearningProgressAlgorithm,
    LearningProgressConfig,
    SingleTaskGenerator,
)


@pytest.fixture
def basic_mg_config():
    """Create basic MettaGrid config for testing."""
    return MettaGridConfig(
        # ... minimal config
    )


def test_single_task_generator_with_mettagrid(basic_mg_config):
    """Test SingleTaskGenerator with MettaGrid config."""
    generator_config = SingleTaskGenerator.Config(env=basic_mg_config)
    generator = SingleTaskGenerator(generator_config)

    task = generator.generate_task("task_0")
    assert task is not None
    assert isinstance(task, MettaGridConfig)


def test_bucketed_generator_with_mettagrid(basic_mg_config):
    """Test BucketedTaskGenerator with MettaGrid config."""
    generator_config = BucketedTaskGenerator.Config.from_mg(basic_mg_config)
    generator = BucketedTaskGenerator(generator_config)

    tasks = generator.get_all_task_ids()
    assert len(tasks) > 0


def test_full_curriculum_with_mettagrid(basic_mg_config):
    """Test full curriculum pipeline with MettaGrid."""
    # Create curriculum
    curriculum_config = CurriculumConfig(
        task_generator=SingleTaskGenerator.Config(env=basic_mg_config),
        algorithm=LearningProgressConfig(),
    )
    curriculum = Curriculum.from_config(curriculum_config)

    # Sample task
    task = curriculum.sample_task()
    assert task is not None

    # Create environment
    env = MettaGridEnv(env_cfg=task.config)
    obs, info = env.reset()
    assert obs is not None
```

**Checklist:**
- [ ] Create `test_mettagrid_integration.py`
- [ ] Add tests for all generator types with MettaGrid
- [ ] Add end-to-end curriculum test
- [ ] Mark as requiring mettagrid: `pytest.importorskip("mettagrid")`

---

### Documentation

#### 19. Create Package Documentation

**Files to create:**

##### `packages/agora/README.md`

```markdown
# Agora: Adaptive Curriculum Learning for RL

Agora provides a flexible, environment-agnostic curriculum learning system for reinforcement learning agents.

## Features

- 🎯 **Adaptive Task Generation**: Automatically generate tasks with varying difficulty
- 📊 **Learning Progress Tracking**: Monitor agent performance across curriculum
- 🔄 **Multiple Algorithms**: Built-in support for learning progress-based curriculum
- 🔌 **Framework Agnostic**: Works with any RL environment (Gymnasium, PettingZoo, etc.)
- ⚡ **Efficient**: Optimized for large-scale distributed training

## Installation

```bash
# Basic installation
pip install agora

# With PufferLib support
pip install agora[puffer]

# With MettaGrid integration
pip install agora[mettagrid]

# All optional dependencies
pip install agora[all]
```

## Quick Start

```python
from agora import Curriculum, SingleTaskGenerator

# Define your task configuration (any pydantic model)
task_config = YourTaskConfig(...)

# Create a simple curriculum
generator = SingleTaskGenerator.Config(env=task_config)
curriculum = Curriculum(task_generator=generator)

# Sample tasks during training
for episode in range(1000):
    task = curriculum.sample_task()
    env.reset(config=task.config)
    # ... training loop ...
    curriculum.update_stats(task.id, episode_reward)
```

## Documentation

- [Getting Started](docs/getting_started.md)
- [API Reference](docs/api_reference.md)
- [Algorithms](docs/algorithms.md)
- [Examples](examples/)

## License

MIT License - see [LICENSE](LICENSE) for details.
```

##### `packages/agora/docs/getting_started.md`

Basic tutorial for new users

##### `packages/agora/docs/api_reference.md`

Full API documentation

##### `packages/agora/examples/basic_curriculum.py`

Working example

**Checklist:**
- [ ] Create `README.md`
- [ ] Create `docs/getting_started.md`
- [ ] Create `docs/api_reference.md`
- [ ] Create `docs/algorithms.md`
- [ ] Create `examples/basic_curriculum.py`
- [ ] Create `examples/learning_progress.py`
- [ ] Create `examples/custom_generator.py`

---

### Final Validation

#### 20. Comprehensive Testing Checklist

**Agora package tests:**
- [ ] Run `cd packages/agora && uv run pytest tests/ -v`
- [ ] Verify all tests pass
- [ ] Check test coverage: `uv run pytest tests/ --cov=agora --cov-report=html`
- [ ] Aim for >80% coverage

**Metta integration tests:**
- [ ] Run `uv run pytest tests/cogworks/curriculum/ -v` (should use shim)
- [ ] Verify deprecation warnings appear
- [ ] Verify all tests still pass

**Full test suite:**
- [ ] Run `uv run pytest tests/ -v` (full metta test suite)
- [ ] Verify no regressions

**Training recipes:**
- [ ] Test arena recipe: `timeout 30s uv run ./tools/run.py experiments.recipes.arena.train run=test`
- [ ] Test cogs_v_clips recipe: `timeout 30s uv run ./tools/run.py experiments.recipes.cogs_v_clips.level_1 run=test`
- [ ] Verify curriculum-based training works

**Code quality:**
- [ ] Run `ruff format packages/agora/`
- [ ] Run `ruff check --fix packages/agora/`
- [ ] Run `mypy packages/agora/src/`
- [ ] Fix all type errors

**Build & install:**
- [ ] Build package: `cd packages/agora && uv build`
- [ ] Test install in fresh venv
- [ ] Verify imports work: `python -c "import agora; print(agora.__version__)"`

---

## Implementation Timeline

### Week 1: Foundation
- **Day 1 AM**: Create package structure, pyproject.toml, config abstraction
- **Day 1 PM**: Migrate task_tracker, shared_memory_backend
- **Day 2 AM**: Migrate stats, lp_scorers
- **Day 2 PM**: Start task_generator refactoring

### Week 2: Core Migration
- **Day 3**: Complete task_generator split, migrate learning_progress
- **Day 4 AM**: Migrate curriculum.py, puffer wrapper
- **Day 4 PM**: Create __init__.py, test basic imports

### Week 3: Integration & Testing
- **Day 5**: Migrate test suite, create integration tests
- **Day 6**: Update metta imports, test with shim
- **Day 7**: Comprehensive testing, fix issues

### Week 4: Documentation & Polish
- **Day 8**: Write documentation, examples
- **Day 9**: Final testing, code review
- **Day 10**: Tag release, build package

---

## Benefits of This Approach

### 1. **Reusability**
- Can be used with any RL environment (not just MettaGrid)
- Clean separation of concerns
- Easier to maintain and test

### 2. **Independent Evolution**
- Package can be versioned independently
- Faster iteration cycles
- Clear API contracts

### 3. **Community Adoption**
- Other projects can use curriculum learning
- Potential for contributions
- Better documentation and examples

### 4. **Better Testing**
- Isolated test suite
- Clear dependency boundaries
- Easier CI/CD

### 5. **Code Quality**
- Cleaner abstractions
- Type safety improvements
- Better documentation

---

## Risks & Mitigation

### Risk 1: Breaking Changes During Migration

**Mitigation**:
- Keep backward compatibility shim
- Extensive testing before removal
- Clear deprecation warnings

### Risk 2: Circular Dependencies

**Mitigation**:
- Use Protocol/ABC for abstractions
- Optional dependencies pattern
- Clear dependency graph

### Risk 3: Performance Regression

**Mitigation**:
- Benchmark before/after
- Profile critical paths
- Optimize generic implementations

### Risk 4: Documentation Drift

**Mitigation**:
- Automated doc generation
- Example code as tests
- Regular doc reviews

---

## Success Metrics

### Technical Metrics
- ✅ All 47 usage sites updated
- ✅ 100% test coverage maintained
- ✅ Zero performance regression
- ✅ Package builds successfully
- ✅ All linters pass

### Quality Metrics
- ✅ Complete API documentation
- ✅ 5+ working examples
- ✅ Migration guide complete
- ✅ CI/CD pipeline green

### Adoption Metrics (post-release)
- External usage (if public PyPI)
- GitHub stars (if public repo)
- Community contributions

---

## Timeline Summary

| Week | Phase | Deliverables |
|------|-------|-------------|
| 1 | Setup & Core Migration | Package structure, pyproject.toml, core code migrated |
| 2 | Testing & Docs | Tests migrated, docs written, examples created |
| 3 | Integration | Metta codebase updated, full testing, validation |
| 4 | Polish & Release | Version tag, build, publish preparation |

**Total estimated time**: 3-4 weeks with 1-2 developers

---

## Next Steps

1. **Review & Approve**: Team reviews this plan
2. **Name Selection**: Confirmed package name: `agora`
3. **Create Tracking Issue**: GitHub issue with checklist
4. **Start Phase 1**: Create package structure
5. **Daily Progress Updates**: Track migration progress

---

## Appendix: Key Files Inventory

### Files to Migrate (9 files)
1. `curriculum.py` - 523 lines - Core curriculum logic
2. `learning_progress_algorithm.py` - 379 lines - LP algorithm
3. `task_generator.py` - 419 lines - Task generation
4. `task_tracker.py` - TaskTracker implementation
5. `stats.py` - Statistics collection
6. `lp_scorers.py` - Learning progress scorers
7. `shared_memory_backend.py` - Multiprocess support
8. `curriculum_env.py` - PufferEnv wrapper
9. `demo.py` - Examples

### Tests to Migrate (12+ files)
- All files in `tests/cogworks/curriculum/`

### Files to Update (47 files)
- All files that import from `metta.cogworks.curriculum`
- See grep results for complete list

---

## Appendix: Example Usage

### Basic Curriculum
```python
from agora import Curriculum, SingleTaskGenerator

# Create task generator with your config
generator = SingleTaskGenerator(config=my_env_config)

# Create curriculum
curriculum = Curriculum(task_generator=generator)

# Training loop
for episode in range(1000):
    task = curriculum.sample_task()
    env.reset(task)
    # ... train ...
    curriculum.update_stats(task.id, episode_stats)
```

### Learning Progress
```python
from agora import Curriculum, LearningProgressAlgorithm, BucketedTaskGenerator

# Create adaptive task generator
generator = BucketedTaskGenerator.from_config(config)

# Create LP curriculum
curriculum = Curriculum(
    task_generator=generator,
    algorithm=LearningProgressAlgorithm(config=lp_config)
)

# Automatically adapts difficulty based on learning progress
for episode in range(1000):
    task = curriculum.sample_task()  # Selects based on LP
    # ... train ...
```

### MettaGrid Integration
```python
from agora import Curriculum
from agora.mettagrid import mettagrid_curriculum  # Optional
from mettagrid import MettaGridEnv, MettaGridConfig

config = MettaGridConfig(...)
curriculum = mettagrid_curriculum(config)

env = MettaGridEnv(env_cfg=config)
task = curriculum.sample_task()
env.reset(task_config=task.config)
```

---

**End of Document**

