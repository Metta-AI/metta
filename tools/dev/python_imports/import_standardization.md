# Python Import Standardization and Enforcement

_For temporary reference during implementation. This file will be deleted._

This work standardizes Python imports to prevent circular dependencies and improve developer experience.

- All files start with `from __future__ import annotations`
- Shared types imported from `types.py` files
- Cross-package imports use module pattern: `import x.y as y_mod`
- Eliminate TYPE_CHECKING workarounds (except legitimate lazy loading)

## 0. Tooling & Documentation

- Analysis tools: `detect_cycles.py`, `analyze_architecture.py`
- Documentation: `SPECIFICATION.md`
- CI validation script (draft): `ci_validate_imports.py`
- Update `CLAUDE.md` and `.cursor/rules/101_python.mdc`

## 1. Type Extraction

Extract shared types to `types.py` files to break circular dependencies at their source

**Priority Packages:**

- `metta/rl/types.py` - RL configs and protocols
- `metta/cogworks/types.py` - Curriculum types
- `metta/sweep/types.py` - Sweep configuration types
- `agent/src/metta/agent/types.py` - Agent component types
- `devops/types.py` - Infrastructure types

## 2. Simplify `__init__.py` files

Remove heavy re-exports from `__init__.py` files

- 4 files flagged as "simplify" (high complexity)
- 3 files flagged as "refactor_lazy_loading" (mixing heavy and light imports)
- 91 files that can be emptied

**For files with lazy loading to refactor:**

- Extract light imports (configs) to types.py
- Keep lazy loading only for heavy modules (torch, ML frameworks)
- Reduce `__all__` to essential symbols

**Special Cases:**

- **Keep:** `metta/rl/training/__init__.py` - legitimate lazy loading (but refactor light exports)
- **Preserve:** Files with `__getattr__` for performance-critical lazy loading

## 3. TYPE_CHECKING Conversion

Convert TYPE_CHECKING imports to module imports or types.py references

- 96 files using TYPE_CHECKING pattern

**If type in types.py:** Replace with direct import

```python
# Before
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from metta.rl.training import TrainingEnvironmentConfig

# After
from __future__ import annotations
from metta.rl.types import TrainingEnvironmentConfig
```

**If circular dependency remains:** Use module import

```python
# Before
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from metta.rl.trainer import Trainer

# After
from __future__ import annotations
import metta.rl.trainer as trainer
# Use as: trainer.Trainer
```

**If legitimate lazy loading:** Keep with documentation

```python
# Keep - documented performance exception
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from expensive_module import HeavyClass
# Used in __getattr__ lazy loading
```

## 4. Validation & Enforcement

Enable automated enforcement of import patterns

**CI Integration**

- Integrate `ci_validate_imports.py` into GitHub Actions
- Start in gradual mode (fail on cross-package cycles only)
- Add to `metta ci` command

**Pre-commit Hooks**

- Check `from __future__ import annotations` present
- Validate import ordering
- Detect new TYPE_CHECKING without justification
- Flag local imports (function-level)

**Ruff Configuration**

```toml
[tool.ruff.lint]
select = ["I", "TCH", "TID"]  # Import rules

[tool.ruff.lint.isort]
required-imports = ["from __future__ import annotations"]
```

**Validation Stages:**

- **Gradual mode (initial):** Fail on cross-package cycles, warn on same-package
- **Strict mode (later):** Fail on any circular dependencies

---

## Resolution Protocol

When encountering circular dependencies or import issues:

### 1. Extract Types → types.py (Preferred)

**When:** Type definitions used across multiple modules **Pattern:** Move to appropriate `types.py` file **Benefit:**
Breaks cycles at architectural level

### 2. Convert to Module Imports

**When:** Runtime dependencies prevent type extraction **Pattern:** `import package.module as module` **Benefit:**
Prevents cycles, keeps imports at top

### 3. Use Forward References

**When:** Simple type-only dependencies **Pattern:** `from __future__ import annotations` + qualified names **Benefit:**
No import needed, just string reference

### 4. Document Legitimate Exceptions

**When:** Performance-critical lazy loading **Pattern:** `__getattr__` in `__init__.py` with docstring **Benefit:**
Makes intentional design explicit

**Never:** Local imports inside functions (unless performance exception with documentation)

---

## Architecture Layers

```
Foundation:     common
Environment:    mettagrid.config → mettagrid.simulator → mettagrid.policy/envs
Agent:          agent (neural components)
RL Core:        metta.rl (training, losses, vecenv)
Applications:   metta.sim, metta.cogworks, recipes
Services:       app_backend, tools
```

**Rule:** Modules may import from their own layer or lower layers, never upward.

**Common types:** Available to all layers via `common/src/metta/common/types.py`

---

## References

**Key Files:**

- `CLAUDE.md` - Dev-friendly guidance (permanent)
- `tools/dev/python_imports/SPECIFICATION.md` - Formal rules (permanent)

**Tools:**

- `detect_cycles.py` - Find circular dependencies
- `analyze_architecture.py` - Recommend type extraction
- `ci_validate_imports.py` - CI validation (Phase 4)
