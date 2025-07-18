# Temporary compatibility layer for metta.rl.functions imports
# This file makes old imports work while functions is now a directory
# This file will be removed when all imports are properly updated

import importlib.util
import sys
from pathlib import Path

# Get the functions directory
functions_dir = Path(__file__).parent / "functions"

# Load the __init__.py from the functions directory
spec = importlib.util.spec_from_file_location("metta.rl.functions_pkg", functions_dir / "__init__.py")
functions_pkg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(functions_pkg)

# Re-export everything from the functions package
for name in dir(functions_pkg):
    if not name.startswith("_"):
        globals()[name] = getattr(functions_pkg, name)

# Clean up namespace
del importlib, sys, Path, functions_dir, spec, functions_pkg, name
