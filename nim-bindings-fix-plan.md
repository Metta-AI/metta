# Nim Bindings Fix: Plan & Impact Assessment

## Executive Summary

**Issue**: Installing `cogames` locally crashes with `OSError` when running `cogames tutorial play`, while PyPI install works fine.

**Root Cause**: Different exception types for different failure modes - only some were being caught.

**Fix**: One-line change to catch `OSError` in the policy discovery code.

**Impact**: Minimal - nim_agents are internal-only and undocumented. The fix makes local installs behave identically to PyPI installs.

---

## Problem Description

### Symptoms
```bash
# PyPI install - works
uv pip install cogames
cogames tutorial play  # OK

# Local install - crashes
uv pip install ./packages/cogames
cogames tutorial play  # OSError: dlopen(.../libnim_agents.dylib): no such file
```

### Why This Happens

| Install Type | What's Included | Exception Type | Was Caught? | Result |
|--------------|-----------------|----------------|-------------|--------|
| PyPI wheel | No `bindings/generated/` folder at all | `ModuleNotFoundError` | Yes | Works |
| Local (editable) | `nim_agents.py` but NO `.dylib` | `OSError` from ctypes | **No** | Crash |

The policy discovery system (`loader.py`) walks through all policy packages and tries to import them. When it encounters `nim_agents.py`, Python loads the file which immediately tries to load a native library via ctypes:

```python
# nim_agents.py line 11
dll = cdll.LoadLibrary(os.path.join(dir, libName))  # Raises OSError if .dylib missing
```

---

## The Fix

### File Changed
`packages/mettagrid/python/src/mettagrid/policy/loader.py`

### Before (line 110)
```python
except (ImportError, AttributeError, TypeError):
    # Skip modules that can't be imported (may have missing dependencies)
    pass
```

### After
```python
except (ImportError, AttributeError, TypeError, OSError):
    # Skip modules that can't be imported (may have missing dependencies)
    # OSError covers ctypes failing to load native libraries (e.g., Nim bindings)
    pass
```

---

## Impact Analysis

### What This Affects
- **Policy discovery** in `mettagrid.policy`, `metta.agent.policy`, `cogames.policy`
- Any module with native bindings that fail to load

### What This Does NOT Affect
- **Mettascope UI** - uses separate code path (`MettascopeRenderer` loads directly, not through policy discovery)
- **Any functional policies** - they will still work normally
- **Error visibility** - real import errors are still logged at module level

### Behavior After Fix

| Scenario | Before | After |
|----------|--------|-------|
| PyPI install | Works | Works (unchanged) |
| Local install | Crashes | Works (nim_agents silently skipped) |
| nim_agents availability | N/A (crash) | Silently unavailable |

### Risk Assessment
- **Low risk** - nim_agents are internal-only, not documented in any README or docs
- **No user-facing features lost** - nim_agents were never available to external users anyway
- **Graceful degradation** - matches existing pattern for other optional dependencies

---

## Alternative Solutions (Not Implemented)

### Option B: Full Cibuildwheel Integration
If nim_agents become user-facing in future:

1. Create `cogames_build.py` custom build backend (like mettagrid's `bazel_build.py`)
2. Add cibuildwheel to `.github/workflows/release-cogames.yml`
3. Compile Nim bindings during wheel build
4. Bundle platform-specific `.dylib/.so/.dll` in wheels

**Pros**: nim_agents would work from PyPI
**Cons**: Significant complexity, maintenance burden, CI time

### Option C: Exclude Bindings from Package
Remove `bindings/generated/` from package entirely:

```toml
# pyproject.toml
[tool.setuptools]
exclude-package-data = { cogames = ["**/bindings/**"] }
```

**Pros**: Clean separation
**Cons**: Breaks if anyone relies on finding bindings in installed package

---

## Verification

### Test Commands
```bash
# Clean environment
cd /path/to/test/dir
uv venv .venv

# Install locally (with fix)
uv pip install /path/to/metta3/packages/cogames --python .venv/bin/python --no-cache

# Verify it works
.venv/bin/cogames tutorial play --help

# Verify bindings state (should show nim_agents.py but no .dylib)
.venv/bin/python -c "
import cogames, os
policy_dir = os.path.dirname(cogames.__file__)
bindings = os.path.join(policy_dir, 'policy/nim_agents/bindings/generated')
print(f'Exists: {os.path.exists(bindings)}')
if os.path.exists(bindings): print(f'Contents: {os.listdir(bindings)}')
"
```

### Expected Output
```
Exists: True
Contents: ['__pycache__', 'nim_agents.py']  # Note: no .dylib
```

---

## Related Components

### Mettascope (Unaffected)
- Location: `packages/mettagrid/nim/mettascope/`
- Build: Compiled by Bazel, bundled via cibuildwheel
- Loading: Direct import in `MettascopeRenderer`, not through policy discovery
- Status: Has `libmettascope.dylib` properly compiled in source tree

### Policy Registry System
- `loader.py` discovers policies from multiple packages
- Uses `@functools.cache` for efficiency
- Catches various exceptions to allow graceful degradation
