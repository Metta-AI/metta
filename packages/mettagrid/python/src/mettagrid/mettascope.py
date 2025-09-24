"""
Mettascope Python bindings wrapper.

This module provides access to the mettascope visualization tool through mettagrid.
Usage:
    import mettagrid.mettascope
    mettagrid.mettascope.init(data_dir, replay)
    mettagrid.mettascope.render(step, replay_step)
"""

import sys
from pathlib import Path

# Find the nim/mettascope/bindings/generated directory
# This is relative to the mettagrid package root
package_root = Path(__file__).parent.parent.parent.parent  # Up to packages/mettagrid
nim_bindings_path = package_root / "nim" / "mettascope" / "bindings" / "generated"

# Add the path to sys.path temporarily to import mettascope2
if nim_bindings_path.exists():
    # Insert at the beginning to ensure it's found first
    sys.path.insert(0, str(nim_bindings_path))

    # Import the mettascope2 module
    import mettascope2

    # Remove the path from sys.path to avoid polluting it
    sys.path.remove(str(nim_bindings_path))

    # Re-export the functions and classes

    def init(replay):
        return mettascope2.init(data_dir=str(package_root / "nim" / "mettascope" / "data"), replay=replay)

    render = mettascope2.render
    Mettascope2Error = mettascope2.Mettascope2Error

    __all__ = ["init", "render", "Mettascope2Error"]
else:
    raise ImportError(f"Could not find mettascope bindings at {nim_bindings_path}")
