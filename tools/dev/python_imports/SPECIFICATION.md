# Python Import Management Specification

---

## 1. Purpose and Scope

This specification defines the formal rules for Python import management in the Metta AI codebase. It serves as the
authoritative reference for tooling, CI validation, and automated refactoring. **Target Audience:** CI systems, linters,
automated refactoring tools, and developers needing detailed technical guidance. **Companion Document:** `CLAUDE.md`
provides practical, user-facing guidance on these rules.

---

## 2. Core Principles

### 2.1 Explicit Over Implicit

**Rule:** All type dependencies MUST be explicitly visible at the module level. **Rationale:** Improves code navigation,
enables better static analysis, and makes circular dependencies visible during development rather than at runtime.

### 2.2 Structure Over Workarounds

**Rule:** Circular dependencies MUST be resolved through architectural changes, not import-time workarounds.
**Rationale:** Import-time workarounds (TYPE_CHECKING, local imports) hide structural problems that should be fixed at
the architecture level.

### 2.3 Performance Exceptions

**Rule:** Import-time optimizations are permitted ONLY when:

- Measurably improving startup time for large applications
- Implemented with explicit lazy-loading patterns (`__getattr__`)
- Documented with clear performance justification **Rationale:** Premature optimization creates complexity; legitimate
  performance needs are met with explicit patterns.

### 2.4 Absolute Over Relative Imports

**Rule:** All imports MUST use absolute paths, not relative imports. **Rationale:** Absolute imports are unambiguous,
work correctly when files are run directly, and provide better error messages when the import system is misconfigured.
They also make refactoring easier and improve code readability for new contributors.

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

### 3.1 Standard Import Pattern

**Required for all new code:**

```python
from __future__ import annotations  # MUST be first import
# Standard library imports
import sys
from pathlib import Path
# Third-party imports
import torch
from omegaconf import DictConfig
# Local imports - types first
from metta.common.types import Action, Observation
from mettagrid.types import GridState
# Local imports - runtime dependencies
import metta.rl.trainer as trainer
from metta.mypackage.sibling_module import HelperClass
```

**Ordering:**

1. `from __future__ import annotations` (if needed)
2. Standard library imports
3. Third-party imports
4. Local package imports (types first, then runtime)
5. Each group separated by a blank line

### 3.2 Forward References

**Rule:** The `from __future__ import annotations` import MUST be used to enable forward references automatically.
**Prohibited:** Manual string literals in type hints (except for recursive types).

```python
# CORRECT
from __future__ import annotations
def process(agent: MettaAgent) -> Action:
    pass
# INCORRECT (manual string literals)
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
sim = simulator.Simulator(...)
```

**Decision criteria:**

- Use symbol imports for types from `types.py` files
- Use symbol imports for most dependencies within a package
- Use module imports when symbol import would create a cycle
- Use module imports for large modules to control namespace
- Always use absolute paths (never relative imports)

### 3.4 Optional Types

**Rule:** Use `Optional[type]` syntax for optional parameters, not `type | None`. **Rationale:** While
`from __future__ import annotations` enables union syntax, the codebase standardizes on `Optional` for consistency and
explicit intent.

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

**Rule:** Files named `types.py` MUST contain only lightweight type definitions. **Permitted contents:**

- Data classes (`@dataclass` or plain classes with only data attributes)
- `TypedDict` definitions
- Type aliases (`MyType = Union[...]`)
- `Enum` classes
- `Protocol` definitions (structural typing interfaces)
- Simple Pydantic `Config` classes (field definitions only, no methods with logic) **Prohibited contents:**
- Classes with `__init__` containing business logic
- Classes inheriting from implementation bases (e.g., `StatsLogger`, `ABC` with concrete methods)
- Classes importing heavy dependencies (torch, pandas, boto3)
- Factory methods that instantiate other classes **Rationale:** types.py files should be safe to import from anywhere
  without pulling in heavy dependencies or creating import order issues. If a "type" requires significant
  implementation, it's not a type—it's a class that belongs in a regular module.

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
# DON'T DO THIS - too much implementation
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

### 5.1 Layer Definitions

```
Layer 0 (Foundation):  common
Layer 1 (Environment): mettagrid.config → mettagrid.simulator → mettagrid.policy
Layer 2 (Agent):       agent
Layer 3 (RL Core):     metta.rl
Layer 4 (Apps):        metta.sim, metta.cogworks, recipes
Layer 5 (Services):    app_backend, tools
```

### 5.2 Layer Import Rules

**Rule:** A module MAY import from:

- Its own layer (same package)
- Any lower layer
- `common.types` (accessible from all layers) **Rule:** A module MUST NOT import from:
- A higher layer
- A different top-level package at the same layer (except through `common`) **Example violations:**

```python
# VIOLATION: mettagrid (Layer 1) importing from agent (Layer 2)
from metta.agent.metta_agent import MettaAgent

# CORRECT: Use types
from metta.common.types import AgentProtocol
```

---

## 6. `__init__.py` Management

### 6.1 Default Pattern: Minimal `__init__.py`

**Rule:** By default, `__init__.py` files SHOULD be empty or contain only essential package initialization.
**Rationale:** Heavy re-exports in `__init__.py` files create hidden circular dependencies and slow imports. **Minimal
pattern:**

```python
"""Package description."""
from __future__ import annotations

# Only if necessary
__version__ = "1.0.0"
```

### 6.2 Legitimate `__init__.py` Exports

**Lazy loading rules:**

- **Heavy imports present** → SHOULD use lazy loading (at minimum for heavy imports)
- **Only light imports** → SHOULD use direct imports (lazy loading adds unnecessary complexity) Heavy imports include:
  torch, gymnasium, pufferlib, jax, tensorflow, pandas, scipy, sqlalchemy, boto3, botocore, transformers, or nn.Module
  classes. See `tools/dev/python_imports/analyze_architecture.py` for the authoritative list. **Permitted patterns:**

1. **Lazy loading for performance:**

```python
# Direct imports for light modules
from mypackage.config import Config
from mypackage.types import Action, Observation
# Lazy loading for heavy modules (imports gymnasium, torch, etc.)
def __getattr__(name: str):
    """Lazy-load expensive modules."""
    if name == "PufferEnv":
        from mypackage.envs.puffer_env import PufferEnv
        globals()["PufferEnv"] = PufferEnv
        return PufferEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

Mixing direct and lazy imports is acceptable - the key rule is that heavy imports should be lazy loaded. 2. **Backward
compatibility:**

```python
# Deprecated imports for backward compatibility
# TODO: Remove in version 2.0
from .new_location import OldClass
```

3. **Public API surface:**

```python
# Explicitly define public API
__all__ = ["PublicClass", "public_function"]

from .internal_module import PublicClass, public_function
```

### 6.3 `__init__.py` Simplification Criteria

**Simplify if:**

- Import complexity > 80% of file (mostly re-exports)
- Exports > 20 symbols
- No lazy loading or performance justification **Keep if:**
- Has `__getattr__` for lazy loading
- Has backward compatibility requirements
- Defines explicit public API surface with clear purpose

---

## 7. TYPE_CHECKING Usage

### 7.1 Restricted Usage

**Rule:** `TYPE_CHECKING` imports are PROHIBITED except in the following cases:

1. **Legitimate lazy loading** where documented performance testing shows measurable improvement
2. **Unavoidable runtime cycles** with clear architectural justification documented in code comments **Pattern
   (discouraged, use only when necessary):**

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

### 7.2 Conversion from TYPE_CHECKING

**Before:**

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mettagrid.simulator.simulator import Simulator

def process(sim: 'Simulator') -> None:
    pass
```

**After (preferred - extract types):**

```python
from __future__ import annotations

from mettagrid.types import Simulator

def process(sim: Simulator) -> None:
    pass
```

**After (alternative - module import):**

```python
from __future__ import annotations

import mettagrid.simulator.simulator as simulator

def process(sim: simulator.Simulator) -> None:
    pass
```

---

## 8. Validation and Enforcement - NOT YET IMPLEMENTED

### 8.1 Automated Checks

**CI MUST validate:**

1. No circular dependencies in runtime imports (TYPE_CHECKING imports excluded)
2. All modules follow layer import rules
3. `__init__.py` files meet simplification criteria
4. `from __future__ import annotations` present when type hints used

### 8.2 Pre-commit Hooks

**Pre-commit SHOULD check:**

1. Import ordering within modified files
2. No new TYPE_CHECKING blocks without justification
3. Module import aliases follow naming conventions

### 8.3 Linting Rules

**Ruff/Flake8 configuration:**

```toml
[tool.ruff.lint]
select = [
    "I",     # isort - import ordering
    "TCH",   # flake8-type-checking - TYPE_CHECKING usage
    "TID",   # flake8-tidy-imports - banned imports
]

[tool.ruff.lint.flake8-type-checking]
strict = true
quote-annotations = false  # We use __future__.annotations

[tool.ruff.lint.isort]
known-first-party = ["metta", "mettagrid"]
required-imports = ["from __future__ import annotations"]
```

---

## 9. Exceptions and Variances

### 9.1 Requesting Exception

**Process:**

1. Document specific use case and why standard patterns don't apply
2. Propose alternative approach
3. Get approval from architecture review
4. Document exception in code with clear comment **Example:**

```python
# ARCHITECTURE EXCEPTION: This module uses TYPE_CHECKING due to
# circular dependency between Simulator and Policy that cannot be
# resolved without breaking backward compatibility with external
# plugins. Approved: 2025-11-17. Revisit in v2.0.
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mettagrid.policy import Policy
```

### 9.2 Handling Existing Violations

When encountering code that violates import rules: **During focused tasks:**

- Fix violations only if directly related to the current task
- Note other violations for future cleanup but don't fix them
- Example: If adding a new method to a class, fix imports in that file only **During refactoring tasks:**
- Fix all violations in the scope of the refactor
- Document significant changes in commit messages **When reviewing code:**
- Flag violations but don't block on pre-existing issues
- New code must follow current rules **Priority order for fixes:**

1. Circular dependencies (blocks import)
2. Layer violations (architectural debt)
3. TYPE_CHECKING misuse (code smell)
4. Import ordering (style)

---

## 10. Tooling Reference

### 10.1 Analysis Tools

**detect_cycles.py:** Finds circular dependencies using Tarjan's algorithm

```bash
python devops/import_refactor/detect_cycles.py --path . --output cycles.json
```

**analyze_architecture.py:** Recommends type extraction and `__init__.py` simplification

```bash
python devops/import_refactor/analyze_architecture.py --path . --output architecture.json
```

---

## 11. Glossary

**Circular Dependency:** Module A imports from Module B, and Module B imports from Module A (directly or transitively).
**Runtime Import:** An import that executes when the module is loaded, creating actual dependencies. **Type-Only
Import:** An import used only for type hints, not runtime behavior (often in TYPE_CHECKING blocks). **Module Import:**
Pattern `import x.y.z as z` - imports the module itself, requires qualified access. **Symbol Import:** Pattern
`from x.y import Z` - imports specific symbol, allows direct access. **Forward Reference:** Using a type name in a hint
before it's defined, enabled by `from __future__ import annotations`. **Layer Violation:** Import from a higher
architectural layer to a lower one, creating dependency inversion.

---

## 12. References

- **PEP 563:** Postponed Evaluation of Annotations (`from __future__ import annotations`)
- **PEP 484:** Type Hints
- **CLAUDE.md:** Practical import guidance (lines 558-630)
