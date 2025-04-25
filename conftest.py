# conftest.py at project root
import sys
from pathlib import Path

# Add ./deps and ./deps/pufferlib to sys.path if not already present
base_dir = Path(__file__).resolve().parent
deps_path = base_dir / "deps"
pufferlib_path = deps_path / "pufferlib"

for path in [str(deps_path), str(pufferlib_path)]:
    if path not in sys.path:
        sys.path.insert(0, path)
