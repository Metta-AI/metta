# STYLE_GUIDE.md

## Core Philosophy

Write **lean, pragmatic code** that trusts both your environment and your readers. Favor clarity through simplicity over defensive programming and excessive documentation.

## Key Principles

### 1. Trust Your Environment

**✅ DO:** Assume known invariants
```python
# We know .ruff.toml exists in our repo
with open(ruff_config_path, "rb") as f:
    config = tomllib.load(f)
```

**❌ DON'T:** Add defensive checks for guaranteed conditions
```python
# Unnecessary existence check
if ruff_config_path.exists():
    with open(ruff_config_path, "rb") as f:
        config = tomllib.load(f)
```

**When to apply:** Internal tooling, controlled environments, known project structure

### 2. Self-Documenting Code

**✅ DO:** Let clear names speak for themselves
```python
def get_world_size(self) -> int:
    return self.config.world_size
```

**❌ DON'T:** Add redundant documentation
```python
def get_world_size(self) -> int:
    """Get the number of processes.

    Returns:
        World size
    """
    return self._world_size
```

**When to comment:**
- Ambiguous return values
- Non-obvious behavior
- Important warnings
- Complex algorithms
- The "why" not the "what"

### 3. Direct and Simple

**✅ DO:** Access attributes directly
```python
return self.config.rank
```

**❌ DON'T:** Add unnecessary indirection
```python
self._rank = config.rank  # Stored elsewhere
return self._rank
```

### 4. Conventional Structure

**✅ DO:** Keep all imports at the top
```python
# At top of file
from pathlib import Path
import tomllib
from .core import run_git_cmd
```

**❌ DON'T:** Use inline imports
```python
def some_function():
    from .core import run_git_cmd  # Avoid this
```

**Exceptions:**
- Circular dependency workarounds (consider this a code smell to fix later)
- Heavy imports in CLI tools (e.g., `torch`, `wandb`) where startup time matters

## General Guidelines

- **Error handling:** Only catch exceptions you can meaningfully handle
- **Defensive programming:** Reserve for truly unpredictable external inputs
- **Code length:** Shorter is better when it doesn't sacrifice clarity
- **Comments:** Should add value, not repeat what the code already says
- **Dependencies:** Make them explicit and visible at the module level

## Summary

Our code should be:
- **Conventional** - Follow established patterns
- **Clean** - Remove unnecessary ceremony
- **Direct** - Don't be clever when simple will do

When in doubt, ask: "What's the simplest thing that could possibly work?" Then write that.
