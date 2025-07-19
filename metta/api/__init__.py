# Temporary compatibility layer for metta.api imports
# This allows `from metta.api import X` to work while api.py is still a single file
# This file will be removed when api.py is properly split into a package

import importlib.util
import sys
from pathlib import Path

# Get the parent directory
parent_dir = Path(__file__).parent.parent

# Import the api module
spec = importlib.util.spec_from_file_location("_api", parent_dir / "api.py")
_api = importlib.util.module_from_spec(spec)
sys.modules["_api"] = _api
spec.loader.exec_module(_api)

# Re-export everything from api.py
for name in dir(_api):
    if not name.startswith("_"):
        globals()[name] = getattr(_api, name)

# Clean up
del sys, Path, parent_dir, importlib, spec, _api, name
