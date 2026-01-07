# Nim Bindings Investigation Report

## Investigation Timeline & Hypotheses

### Initial Hypothesis (Incorrect)
> "PyPI wheel doesn't include bindings, local install does"

**Testing revealed**: Both include bindings differently, but the failure mode differs.

### Hypothesis 2 (Partially Correct)
> "The .dylib is platform-specific and not being built for macOS"

**Finding**: True, but the real issue is that local installs include the Python wrapper but not the compiled library.

### Final Root Cause
> "Local install includes `nim_agents.py` (Python ctypes wrapper) but NOT `libnim_agents.dylib` (the native library). PyPI wheel includes neither. The exception handler only caught `ImportError`, not `OSError` from ctypes."

---

## Useful Commands

### Package Installation & Management

```bash
# Install from PyPI
uv pip install cogames --python .venv/bin/python

# Install from local source
uv pip install /path/to/packages/cogames --python .venv/bin/python --no-cache

# Uninstall
uv pip uninstall cogames mettagrid --python .venv/bin/python

# Download wheel without installing (for inspection)
uv pip download cogames --python-version 3.12 --python-platform macosx_14_0_arm64 -d ./wheels
```

### Wheel Inspection

```bash
# List wheel contents
unzip -l wheels/cogames-*.whl | head -50

# Extract and inspect
unzip wheels/cogames-*.whl -d ./extracted
ls -la extracted/cogames/policy/nim_agents/

# Check for bindings in wheel
unzip -l wheels/cogames-*.whl | grep -i bindings
```

### Installed Package Inspection

```bash
# Find where package is installed
.venv/bin/python -c "import cogames; print(cogames.__file__)"

# Check bindings directory state
.venv/bin/python -c "
import cogames, os
policy_dir = os.path.dirname(cogames.__file__)
bindings = os.path.join(policy_dir, 'policy/nim_agents/bindings/generated')
print(f'Exists: {os.path.exists(bindings)}')
if os.path.exists(bindings): print(f'Contents: {os.listdir(bindings)}')
"

# Check mettagrid installation
.venv/bin/python -c "import mettagrid; print(mettagrid.__file__)"
```

### Testing Commands

```bash
# Test basic functionality
.venv/bin/cogames tutorial play --help

# Test with timeout (for interactive commands)
timeout 5 .venv/bin/cogames tutorial play 2>&1 || echo "Exited (expected)"

# Run specific test
uv run pytest packages/cogames/tests/test_policies.py -v
```

### Git & Diff Commands

```bash
# Check what's changed
git diff packages/mettagrid/python/src/mettagrid/policy/loader.py

# Show only modified files
git status --short

# Stash changes for clean testing
git stash
git stash pop
```

---

## Key Files & Their Roles

### Policy Discovery System

| File | Role |
|------|------|
| `packages/mettagrid/python/src/mettagrid/policy/loader.py` | Main policy discovery, THE FIX LOCATION |
| `packages/mettagrid/python/src/mettagrid/policy/policy_registry.py` | Policy name → class path mapping |
| `packages/mettagrid/python/src/mettagrid/policy/policy.py` | Base policy classes |

### Cogames Nim Agents

| File | Role |
|------|------|
| `packages/cogames/src/cogames/policy/nim_agents/nim_agents.nim` | Main Nim source |
| `packages/cogames/src/cogames/policy/nim_agents/bindings/generated/nim_agents.py` | Generated Python ctypes wrapper |
| `packages/cogames/src/cogames/policy/nim_agents/bindings/generated/libnim_agents.dylib` | Compiled native library (MISSING in installs) |
| `packages/cogames/setup.py` | Contains `_build_nim()` for compilation |

### Mettagrid Mettascope (For Comparison)

| File | Role |
|------|------|
| `packages/mettagrid/nim/mettascope/src/mettascope.nim` | Main Nim source |
| `packages/mettagrid/nim/mettascope/bindings/generated/libmettascope.dylib` | Compiled native library (EXISTS) |
| `packages/mettagrid/python/src/mettagrid/renderer/mettascope.py` | Python wrapper with `_resolve_nim_root()` |
| `packages/mettagrid/bazel_build.py` | Custom build backend that compiles everything |

### Package Configuration

| File | Role |
|------|------|
| `packages/cogames/pyproject.toml` | Package config, package-data settings |
| `packages/mettagrid/pyproject.toml` | Uses custom `bazel_build` backend |
| `.github/workflows/release-cogames.yml` | Simple `python -m build` |
| `.github/workflows/release-mettagrid.yml` | Uses cibuildwheel |

---

## Key Configuration Snippets

### cogames pyproject.toml
```toml
[build-system]
requires = ["setuptools==80.9.0", "wheel==0.45.1", "setuptools_scm==8.1.0"]
build-backend = "setuptools.build_meta"  # Standard backend

[tool.setuptools.package-data]
cogames = ["py.typed", "maps/*.map", "maps/**/*.map", "assets/**/*"]
# Note: No bindings/*.dylib included!

[tool.setuptools]
include-package-data = true  # This causes .py files to be included but not .dylib
```

### mettagrid pyproject.toml (for comparison)
```toml
[build-system]
requires = [...]
build-backend = "bazel_build"  # Custom backend!
backend-path = ["."]

[tool.setuptools.package-data]
"mettagrid" = ["py.typed", "*.so", "*.pyi", "*.pyd", "*.dylib", "nim/mettascope/**/*"]
# Note: Explicitly includes *.dylib
```

### loader.py exception handling (THE FIX)
```python
# Before
except (ImportError, AttributeError, TypeError):
    pass

# After
except (ImportError, AttributeError, TypeError, OSError):
    # OSError covers ctypes failing to load native libraries
    pass
```

---

## Investigation Findings

### Finding 1: PyPI Wheel Structure
```
wheels/cogames-0.3.41-py3-none-any.whl contents:
  cogames/
  cogames/policy/
  cogames/policy/nim_agents/
  cogames/policy/nim_agents/__init__.py
  cogames/policy/nim_agents/random.py
  cogames/policy/nim_agents/thinky.py
  # NO bindings/generated/ folder at all!
```

### Finding 2: Local Install Structure
```
site-packages/cogames/policy/nim_agents/bindings/generated/
  __pycache__/
  nim_agents.py      # EXISTS - Python ctypes wrapper
  # NO libnim_agents.dylib!
```

### Finding 3: Why the Difference
- `include-package-data = true` in pyproject.toml
- This includes `.py` files that are in the source tree
- But `.dylib` files are not included (not in MANIFEST.in, not explicitly listed)
- PyPI build runs in clean environment → no bindings folder exists → nothing to include
- Local build uses source tree → `bindings/generated/nim_agents.py` exists → gets included

### Finding 4: Exception Flow
```
PyPI install:
  loader.py → importlib.import_module("cogames.policy.nim_agents.bindings.generated")
  → ModuleNotFoundError (no such module)
  → Caught by except (ImportError, ...)
  → Silently skipped ✓

Local install:
  loader.py → importlib.import_module("cogames.policy.nim_agents.bindings.generated.nim_agents")
  → Module exists, Python loads nim_agents.py
  → nim_agents.py line 11: cdll.LoadLibrary("libnim_agents.dylib")
  → OSError: dlopen(...): no such file
  → NOT caught by except (ImportError, ...)
  → Crash ✗
```

### Finding 5: Mettascope Works Differently
```python
# mettagrid/renderer/mettascope.py
def _resolve_nim_root():
    # Checks source tree first, then packaged location
    source = _python_package_root.parent.parent.parent / "nim" / "mettascope"
    packaged = _python_package_root / "nim" / "mettascope"
    for root in [source, packaged]:
        if (root / "bindings" / "generated").exists():
            return root
    return None
```
- Mettascope looks for pre-compiled dylib in known locations
- Falls back gracefully if not found
- Bazel compiles the dylib in source tree during development
- Cibuildwheel bundles it properly for PyPI

---

## Files Created During Investigation

| File | Purpose |
|------|---------|
| `/Users/yatharth/Documents/softmax/onboarding2/.venv/` | Clean test environment |
| `wheels/` | Downloaded wheels for inspection |
| `extracted/` | Extracted wheel contents |

---

## Reproduction Steps

### To Reproduce the Bug (Before Fix)
```bash
# 1. Create clean environment
mkdir test-dir && cd test-dir
uv venv .venv

# 2. Stash the fix if applied
cd /path/to/metta3
git stash

# 3. Install locally
uv pip install ./packages/cogames --python /path/to/test-dir/.venv/bin/python --no-cache

# 4. Trigger the crash
/path/to/test-dir/.venv/bin/cogames tutorial play
# Expected: OSError about libnim_agents.dylib
```

### To Verify the Fix
```bash
# 1. Apply the fix to loader.py
# 2. Uninstall and reinstall
uv pip uninstall cogames mettagrid --python .venv/bin/python
uv pip install ./packages/cogames --python .venv/bin/python --no-cache

# 3. Verify it works
.venv/bin/cogames tutorial play --help
# Expected: Help text, no crash
```

---

## Lessons Learned

1. **Exception handling matters**: Different failure modes produce different exceptions. `ImportError` vs `OSError` vs `ModuleNotFoundError` all behave differently.

2. **Package inclusion is tricky**: `include-package-data = true` doesn't include everything - only files matching certain patterns and present in the source tree.

3. **Local vs CI builds differ**: CI builds in clean environments, so stale artifacts don't exist. Local builds use the source tree which may have partial build artifacts.

4. **Test both install methods**: Always test both PyPI install and local install when dealing with native extensions.
