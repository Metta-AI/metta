#!/usr/bin/env python3
import importlib.util
import os
import sys

# Determine which build system to use based on environment variable or command line argument
# Default to cython if not specified
build_system = os.environ.get("BUILD_SYSTEM", "cython").lower()

# Allow command line override with --build-system=pybind or --build-system=cython
for arg in sys.argv:
    if arg.startswith("--build-system="):
        build_system = arg.split("=")[1].lower()
        sys.argv.remove(arg)
        break

# Map build system names to their setup file
setup_files = {
    "cython": "setup-cython.py",
    "pybind": "setup-pybind.py",
}

if build_system not in setup_files:
    print(f"Error: Unknown build system '{build_system}'. Valid options are: {', '.join(setup_files.keys())}")
    sys.exit(1)

setup_file = setup_files[build_system]
print(f"Using build system: {build_system} (from {setup_file})")

# Import and run the appropriate setup file as a module
spec = importlib.util.spec_from_file_location("setup_module", setup_file)
if spec is None or spec.loader is None:
    print(f"Error: Could not find {setup_file}")
    sys.exit(1)

setup_module = importlib.util.module_from_spec(spec)
sys.modules["setup_module"] = setup_module
spec.loader.exec_module(setup_module)
