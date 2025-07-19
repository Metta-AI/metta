# Import Fix for metta.rl.functions

## Problem
If you're seeing this error:
```
FileNotFoundError: [Errno 2] No such file or directory: '/workspace/metta/metta/rl/functions.py'
```

This means your local branch has an outdated state where Python is looking for `metta/rl/functions.py` (a file) instead of `metta/rl/functions/` (a directory).

## Quick Fix
Run the fix script:
```bash
./fix_imports.sh
```

## Manual Fix
If the script doesn't work, manually fix it:

1. **Pull latest changes:**
   ```bash
   git pull origin richard-func-to-folder
   ```

2. **Remove any leftover functions.py file:**
   ```bash
   rm -f metta/rl/functions.py
   ```

3. **Clean Python cache:**
   ```bash
   find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
   find . -name "*.pyc" -delete 2>/dev/null
   ```

4. **Verify the structure:**
   ```bash
   ls -la metta/rl/functions/
   # Should show __init__.py and other .py files
   ```

## Diagnostics
To debug import issues, run:
```bash
python debug_imports.py
```

This will show:
- Current git commit
- Python path configuration
- File system state
- Module resolution details
- Import test results

## What Changed
- `metta/rl/functions.py` (single file) → `metta/rl/functions/` (directory)
- All functions are now organized into submodules:
  - `advantage.py`
  - `policy_management.py`
  - `rollout.py`
  - `stats.py`
  - `training.py`
- The `__init__.py` re-exports all functions for backward compatibility
