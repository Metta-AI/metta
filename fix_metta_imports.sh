#!/bin/bash
# fix_metta_imports.sh
# Consolidated script to resolve all metta.rl.functions import issues.
# 1. Pull latest branch
# 2. Remove stale functions.py files and .pyc caches in *all* namespace paths
# 3. Remove any installed wheels/editable installs of metta* packages in the venv
# 4. Verify that import now works

set -euo pipefail

BRANCH="richard-func-to-folder"

printf "\n=== Metta import-fix script ===\n\n"

# Ensure we’re at repo root
if [ ! -d ".git" ]; then
  echo "Run this from the repository root" >&2
  exit 1
fi

# Pull latest code
printf "Updating branch…\n"
git fetch origin "$BRANCH"
git checkout "$BRANCH"
git pull --ff-only origin "$BRANCH"

echo
printf "Removing any stray functions.py files…\n"
find . -path "*/metta/rl/functions.py" -exec rm -f {} +

echo
printf "Removing cached byte-code in every metta namespace path…\n"
PYTHON="python"
$PYTHON - <<'PY'
import sys, importlib.util, pathlib, subprocess, os
import pkg_resources  # type: ignore

paths = []
try:
    import metta
    paths.extend(list(metta.__path__))
except ImportError:
    pass
# Also look in site-packages for any installed wheels
for dist in pkg_resources.working_set:
    if dist.project_name.startswith("metta"):
        paths.append(dist.location)

for p in paths:
    if not os.path.isdir(p):
        continue
    # Delete any functions*.pyc in rl/__pycache__
    for root, dirs, files in os.walk(p):
        if root.endswith(os.sep + "rl" + os.sep + "__pycache__"):
            for f in files:
                if f.startswith("functions") and f.endswith(".pyc"):
                    try:
                        os.remove(os.path.join(root, f))
                        print("Removed", os.path.join(root, f))
                    except Exception:
                        pass
# Also remove any empty __pycache__ dirs afterwards
for p in paths:
    for root, dirs, _ in os.walk(p):
        for d in dirs:
            if d == "__pycache__":
                full = os.path.join(root, d)
                try:
                    os.rmdir(full)
                except OSError:
                    pass
PY

echo
printf "Uninstalling any installed metta* distributions…\n"
pip uninstall -yq metta metta-rl metta-common metta-mettagrid || true
# Clean residual dir
site_dir=$(python - <<'PY'
import site, json, sys, pathlib
paths = site.getsitepackages()
print(paths[0] if paths else "")
PY)
if [ -n "$site_dir" ]; then
  rm -rf "$site_dir/metta" 2>/dev/null || true
  rm -f  "$site_dir"/__editable__.*metta*.pth 2>/dev/null || true
fi

echo
printf "Verifying structure…\n"
if [ ! -f "metta/rl/functions/__init__.py" ]; then
  echo "ERROR: metta/rl/functions/__init__.py missing" >&2
  exit 1
fi

printf "Testing import…\n"
python - <<'PY'
try:
    from metta.rl.functions import cleanup_old_policies
    print("✅ Import succeeded – environment fixed")
except Exception as e:
    print("❌ Import still failing:", e)
    raise SystemExit(1)
PY

echo
printf "=== Fix complete ===\n"
