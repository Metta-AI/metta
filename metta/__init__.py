# metta/__init__.py
import sys
from pathlib import Path

# Get the two potential metta source directories
metta_src_dir = str(Path(__file__).parent.resolve())
mettagrid_src_dir = str((Path(__file__).parent.parent / "mettagrid/src/metta").resolve())

# Prioritize the main metta source directory
# This ensures that imports like `metta.rl` resolve to the correct package,
# not a conflicting module in the mettagrid source tree.
if metta_src_dir in sys.path:
    sys.path.remove(metta_src_dir)
sys.path.insert(0, metta_src_dir)

if mettagrid_src_dir in sys.path:
    sys.path.remove(mettagrid_src_dir)
sys.path.append(mettagrid_src_dir)  # Add it to the end to deprioritize it


# Now, extend the path for namespace packaging
__path__ = __import__("pkgutil").extend_path(__path__, __name__)
