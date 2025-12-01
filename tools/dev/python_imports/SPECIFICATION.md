# Python Import Management Specification

---

## 1. Purpose and Scope

This specification defines the formal rules for Python import management in the Metta AI codebase. It serves as the
authoritative reference for tooling, CI validation, and automated refactoring.

**Target Audience:** CI systems, linters, automated refactoring tools, and developers needing detailed technical
guidance.

**Companion Document:** `CLAUDE.md` provides practical, user-facing guidance on these rules.

---

## 2. Core Principles

### 2.1 Explicit Over Implicit

**Rule:** All type dependencies MUST be explicitly visible at the module level.

**Rationale:** Improves code navigation, enables better static analysis, and makes circular dependencies visible during
development rather than at runtime.

### 2.2 Structure Over Workarounds

**Rule:** Circular dependencies MUST be resolved through architectural changes, not import-time workarounds.
**Rationale:** Import-time workarounds (TYPE_CHECKING, local imports) hide structural problems that should be fixed at
the architecture level.

### 2.3 Performance Exceptions

**Rule:** Import-time optimizations are permitted ONLY when:

- Improving startup time for large applications
- Implemented with explicit lazy-loading patterns (`__getattr__`)
- Documented with clear performance justification

**Rationale:** Premature optimization creates complexity; legitimate performance needs are met with explicit patterns.

### 2.4 Absolute Over Relative Imports

**Rule:** All imports MUST use absolute paths, not relative imports.

**Rationale:** Absolute imports are unambiguous, work correctly when files are run directly, and provide better error
messages when the import system is misconfigured. They also make refactoring easier and improve code readability for new
contributors.

```python
# CORRECT - absolute imports
from metta.cogworks.curriculum.types import CurriculumTask
from metta.cogworks.curriculum.stats import StatsLogger
# INCORRECT - relative imports
from .types import CurriculumTask
from ..common.types import Config
```

---

## 3. Import Patterns

### 3.1 Forward References

**Recommended:** When forward references are needed, use `from __future__ import annotations` to enable them
automatically.

**Needed:**

- Class methods returning the class type: `def clone(self) -> MyClass:`
- Type hints that reference classes defined later in the same file
- Breaking circular import dependencies in type hints

**NOT needed:**

- All types are already imported or defined before use
- Python 3.10+ code using native union syntax (`int | str`)
- Code that performs runtime introspection of `__annotations__`

**Important:** Python 3.14+ (PEP 749) will deprecate `from __future__ import annotations`. The new default behavior will
use lazy evaluation without stringifying annotations. Use this import only when required.

**Pattern:**

```python
# When forward reference is needed
from __future__ import annotations

class Node:
    def get_parent(self) -> Node:  # Forward reference to Node
        pass

def process(agent: MettaAgent) -> Action:  # MettaAgent defined later
    pass
```

**Discouraged:** Manual string literals in type hints (use `from __future__ import annotations` instead).

```python
# DISCOURAGED
def process(agent: 'MettaAgent') -> 'Action':
    pass
```

### 3.3 Symbol vs Module Imports

**Symbol imports (preferred for most cases):**

```python
from metta.common.types import Action
from metta.sim.config import SimConfig
# Usage
action = Action(...)
```

**Module imports (required for circular dependencies):**

```python
import metta.rl.trainer as trainer
import mettagrid.simulator.simulator as simulator
# Usage
def create_simulation():
    sim = simulator.Simulator(...)
    return sim
```

**Decision criteria:**

- Use symbol imports for types from `types.py` files
- Use symbol imports for most dependencies within a package
- Use module imports when symbol import would create a cycle
- Use module imports for modules with many exports to avoid namespace pollution
- Always use absolute paths (never relative imports)

### 3.4 Optional Types

**Rule:** Use `Optional[type]` syntax for optional parameters, not `type | None`.

**Rationale:** Consistent with codebase standard per `CLAUDE.md`.

```python
# CORRECT
from typing import Optional

def process(config: Optional[DictConfig] = None) -> None:
    pass

# INCORRECT
def process(config: DictConfig | None = None) -> None:
    pass
```

---

## 4. Circular Dependency Resolution Protocol

### 4.1 Resolution Hierarchy

When a circular dependency is detected, resolve in this order:

1. **Extract shared types** → Move to appropriate `types.py`
2. **Convert to module imports** → Use module-level imports instead of symbol imports
3. **Refactor shared code** → Move runtime dependencies to lower architectural layer
4. **TYPE_CHECKING (last resort)** → Only for unavoidable runtime cycles with legitimate architectural justification

### 4.2 Type Extraction Rules

**Cross-package types:**

```
common/src/metta/common/types.py
```

**Mettagrid-specific types:**

```
packages/mettagrid/python/src/mettagrid/types.py
```

**Package-specific types:**

```
<package>/types.py
```

**Criteria for extraction:**

- Type is used in 2+ modules within a package
- Type is part of a package's public API
- Type is referenced in type hints (not just runtime)

### 4.3 Types File Content Guidelines

**Rule:** Files named `types.py` MUST contain only lightweight type definitions.

**Permitted contents:**

- Data classes (`@dataclass` or plain classes with only data attributes)
- `TypedDict` definitions
- Type aliases (`MyType = Union[...]`)
- `Enum` classes
- `Protocol` definitions (structural typing interfaces)
- Simple Pydantic `Config` classes (field definitions only, no methods with logic)

**Prohibited contents:**

- Classes with `__init__` containing business logic
- Classes inheriting from implementation bases (e.g., `StatsLogger`, `ABC` with concrete methods)
- Classes importing heavy dependencies (torch, pandas, boto3)
- Factory methods that instantiate other classes

**Rationale:** types.py files should be safe to import from anywhere without pulling in heavy dependencies or creating
import order issues. If a "type" requires significant implementation, it's not a type—it's a class that belongs in a
regular module.

**Example - CORRECT types.py:**

```python
from __future__ import annotations
from typing import Any, Dict, Optional

class CurriculumTask:
    """Simple data holder."""
    def __init__(self, task_id: int, env_cfg, slice_values: Optional[Dict[str, Any]] = None):
        self._task_id = task_id
        self._env_cfg = env_cfg
        self._slice_values = slice_values or {}
```

**Example - INCORRECT types.py:**

```python
# DISCOURAGED - includes excessive implementation
class CurriculumAlgorithm(StatsLogger, ABC):
    def __init__(self, num_tasks: int, hypers: Config):
        # Business logic here...
        StatsLogger.__init__(self, ...)
        self.slice_analyzer = SliceAnalyzer(...)  # Heavy dependency
```

**Alternative resolution** If you can't extract to types.py without making it heavy, look for other ways to break the
cycle:

- Use base class type instead of Union (avoids importing subclasses)
- Use module imports instead of symbol imports
- Restructure to separate interface from implementation

### 4.4 Module Import Conversion

**Pattern:**

```python
# Before (creates cycle)
from mettagrid.simulator.simulator import Simulator

# After (breaks cycle)
import mettagrid.simulator.simulator as simulator
# Usage: simulator.Simulator(...)
```

**Alias naming convention:**

- Use the module's base name as the alias
- Maintain consistency within a package
- Document why module import is needed

---

## 5. Architecture Layers

### 5.1 Layer Import Rules

**Rule:** A module MAY import from:

- Its own layer (same package)
- Any lower layer
- `types` or `common`

**Rule:** A module MUST NOT import from:

- A higher layer
- A different top-level package at the same layer (except through `types` or `common`)

**Example: see packages/mettagrid/pyproject.toml**

```python
# VIOLATION - mettagrid doesn't list metta.agent in dependencies
from metta.agent.metta_agent import MettaAgent

# CORRECT - use shared types that don't require dependency
from metta.common.types import AgentProtocol
```

---

## 6. `__init__.py` Management

### 6.1 Public vs Internal Packages

**Rule:** `__init__.py` design differs based on whether the package is user-facing or internal.

**Public Packages** (in `packages/` directory):

- **Examples:** `mettagrid/__init__.py`, `cortex/__init__.py`, `cogames/__init__.py`
- **Purpose:** Define clean external API for library users
- **Should have:** Thoughtful exports with lazy loading for heavy imports
- **Benefits:**
  - Users import with convenient syntax: `from mettagrid import Simulator`
  - Protects users from internal refactoring (move implementation without breaking API)
  - Defines clear public API surface
  - Better for package documentation

**Internal Modules** (in `metta/` directory):

- **Examples:** `metta/rl/loss/__init__.py`, `metta/cogworks/curriculum/__init__.py`, `metta/sim/envs/__init__.py`
- **Purpose:** Python package organization (implementation detail)
- **Should be:** Empty or minimal
- **Rationale:**
  - No external users - only internal codebase imports
  - VS Code autocompletes full paths: `from metta.rl.loss.ppo import PPOLoss`
  - Avoids hidden circular dependencies from re-export chains
  - Makes refactoring easier (no re-export lists to update)
  - Import statements show exactly where code comes from
  - Slightly longer imports don't matter for internal code

### 6.2 Public Package `__init__.py` Pattern

**For packages in `packages/` directory:**

**Lazy loading rules:**

- **Heavy imports present** → MUST use lazy loading for heavy imports
- **Only light imports** → MAY use direct imports (lazy loading adds unnecessary complexity)

Heavy imports include: torch, gymnasium, pufferlib, jax, tensorflow, pandas, scipy, sqlalchemy, boto3, botocore,
transformers, or nn.Module classes. See `tools/dev/python_imports/analyze_architecture.py` for the authoritative list.

**Pattern:**

```python
# mettagrid/__init__.py
from __future__ import annotations

# Direct imports for light modules
from mettagrid.config import GridConfig
from mettagrid.types import Action, Observation

if TYPE_CHECKING:
    from mettagrid.simulator.simulator import Simulator
    from mettagrid.envs.puffer_env import PufferEnv

# Lazy loading for heavy modules (imports gymnasium, torch, etc.)
def __getattr__(name: str):
    """Lazy-load expensive modules."""
    if name == "Simulator":
        from mettagrid.simulator.simulator import Simulator
        globals()["Simulator"] = Simulator
        return Simulator
    if name == "PufferEnv":
        from mettagrid.envs.puffer_env import PufferEnv
        globals()["PufferEnv"] = PufferEnv
        return PufferEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["GridConfig", "Action", "Observation", "Simulator", "PufferEnv"]
```

Mixing direct and lazy imports is acceptable - the key rule is that heavy imports should be lazy loaded.

**Additional permitted patterns:**

1. **Backward compatibility:**

```python
# Deprecated imports for backward compatibility
# TODO: Remove in version 2.0
from mettagrid.new_location import OldClass
```

2. **Explicit public API:**

```python
__all__ = ["PublicClass", "public_function"]

from mettagrid.internal_module import PublicClass, public_function
```

### 6.3 Internal Module `__init__.py` Pattern

**For modules in `metta/` directory:**

**Default pattern (strongly preferred):**

Empty file, or:

```python
"""Package description."""

# Empty - users import from specific modules
```

**When to add exports (rare):**

Only if there's a specific justification:

- Backward compatibility requirement (document reason)
- Legitimate performance benefit with lazy loading

**Anti-pattern to avoid:**

```python
# DON'T DO THIS in internal modules
from metta.rl.loss import PPOLoss, VTraceLoss
from metta.rl.trainer import Trainer
from metta.rl.vecenv import VecEnv
# ... 20+ re-exports

# Instead: Let users import from specific modules
# from metta.rl.loss.ppo import PPOLoss
```

**Simplification criteria:**

Simplify an existing `__init__.py` if:

- Import complexity > 80% of file (mostly re-exports)
- Exports > 20 symbols
- No lazy loading or performance justification
- Not a public package in `packages/`

### 6.4 Module-level Lazy Loading (Non-`__init__.py`)

**Rule:** The `__getattr__` lazy loading pattern MAY be used in regular module files (not just `__init__.py`) when there
is a meaningful performance benefit from deferring heavy imports.

**Use case:** When a module conditionally needs heavy dependencies (torch, pandas, boto3, etc.) but many import paths
don't use those features, lazy loading can significantly improve CLI and import performance.

**Pattern:**

```python
# system_config.py
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

def __getattr__(name: str) -> Any:
    """Lazy-load torch to avoid import cost for CLI scripts that don't need it.
    """
    if name == "torch":
        import torch as _torch
        globals()["torch"] = _torch
        return _torch
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# Rest of module code that uses `torch` variable
def create_optimizer(params) -> torch.optim.Optimizer:
    return torch.optim.Adam(params)  # torch loads on first use
```

**Requirements:**

1. Use TYPE_CHECKING for type hints of lazily-loaded symbols
2. Only apply to heavy dependencies (torch, pandas, scipy, boto3, etc.)
3. Must use `__getattr__` pattern (not inline/local imports inside functions)

Ideally, document the performance benefit in a comment or docstring.

**Rationale:** This is a legitimate Python feature (PEP 562) that allows performance optimization without sacrificing
type safety or creating hidden circular dependencies. Unlike inline imports, it keeps imports at the module level where
they're discoverable.

---

## 7. TYPE_CHECKING Usage

### 7.1 Restricted Usage

**Rule:** `TYPE_CHECKING` imports are PROHIBITED except in the following cases:

1. **Legitimate lazy loading** (see §6.2 for `__init__.py`, §6.4 for regular modules) for performance improvement
2. **Unavoidable runtime cycles** with clear architectural justification

**Pattern (discouraged, use only when necessary):**

```python
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from metta.rl.trainer import Trainer  # Import only for type checking

def configure_trainer(trainer: Trainer) -> None:
    """Configure trainer.

    Note: Uses TYPE_CHECKING due to [specific architectural constraint].
    Consider refactoring to eliminate this pattern.
    """
    pass
```

---

## 8. Tooling Reference

### 8.1 Analysis Tools

**detect_cycles.py:** Finds circular dependencies using Tarjan's algorithm

```bash
python tools/dev/python_imports/detect_cycles.py --path . --output cycles.json
```

**analyze_architecture.py:** Recommends type extraction and `__init__.py` simplification

```bash
python tools/dev/python_imports/analyze_architecture.py --path . --output architecture.json
```

---

## 9. Glossary

**Circular Dependency:** Module A imports from Module B, and Module B imports from Module A (directly or transitively).

**Runtime Import:** An import that executes when the module is loaded, creating actual dependencies.

**Type-Only Import:** An import used only for type hints, not runtime behavior (often in TYPE_CHECKING blocks).

**Module Import:** Pattern `import x.y.z as z` - imports the module itself, requires qualified access.

**Symbol Import:** Pattern `from x.y import Z` - imports specific symbol, allows direct access.

**Forward Reference:** Using a type name in a hint before it's defined, requiring either
`from __future__ import annotations` (Python 3.7-3.13) or lazy evaluation (Python 3.14+).

**Layer Violation:** Import from a higher architectural layer to a lower one, creating dependency inversion.

---

## 10. References

- **PEP 484:** Type Hints
- **PEP 563:** Postponed Evaluation of Annotations (`from __future__ import annotations`)
- **PEP 749:** Lazy Evaluation of Annotations (Python 3.14+, deprecates PEP 563)
- **CLAUDE.md:** Practical import guidance (lines 558-630)
