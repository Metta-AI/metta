#!/usr/bin/env python
"""Diagnostic script to debug metta.rl.functions import issues."""

import importlib.util
import os
import subprocess
import sys

print("=== Import Diagnostics for metta.rl.functions ===\n")

# Check current directory
print(f"Current directory: {os.getcwd()}")

# Check git status
print("\n--- Git Status ---")
try:
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    print(f"Current commit: {result.stdout.strip()}")

    result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    if result.stdout:
        print("Uncommitted changes:")
        print(result.stdout)
    else:
        print("No uncommitted changes")
except Exception as e:
    print(f"Error checking git status: {e}")

# Check Python path
print("\n--- Python Path ---")
for i, path in enumerate(sys.path):
    if "metta" in path or path == "":
        print(f"{i}: {path}")

# Check for functions.py file
print("\n--- File System Check ---")
functions_py = os.path.join("metta", "rl", "functions.py")
functions_dir = os.path.join("metta", "rl", "functions")
functions_init = os.path.join("metta", "rl", "functions", "__init__.py")

print(f"functions.py exists: {os.path.exists(functions_py)}")
print(f"functions/ directory exists: {os.path.isdir(functions_dir)}")
print(f"functions/__init__.py exists: {os.path.exists(functions_init)}")

# Check for cached files
print("\n--- Cache Check ---")
if os.path.exists("metta/rl/__pycache__"):
    pycache_files = os.listdir("metta/rl/__pycache__")
    functions_cache = [f for f in pycache_files if "functions" in f and ".pyc" in f]
    if functions_cache:
        print(f"Found cached files: {functions_cache}")
    else:
        print("No functions-related cache files found")
else:
    print("No __pycache__ directory")

# Try to find the module
print("\n--- Module Resolution ---")
try:
    spec = importlib.util.find_spec("metta.rl.functions")
    if spec:
        print(f"Module found at: {spec.origin}")
        print(f"Is package: {spec.submodule_search_locations is not None}")
        if spec.submodule_search_locations:
            print(f"Package location: {spec.submodule_search_locations}")
    else:
        print("Module not found by importlib")
except Exception as e:
    print(f"Error finding module: {e}")

# Try to import and see what happens
print("\n--- Import Test ---")
try:
    import metta.rl.functions

    print("✓ Import successful")
    print(f"Module file: {metta.rl.functions.__file__}")
    print(f"Module path: {getattr(metta.rl.functions, '__path__', 'Not a package')}")
except FileNotFoundError as e:
    print(f"✗ FileNotFoundError: {e}")
    print(f"Python is looking for: {e.filename}")
except Exception as e:
    print(f"✗ Other error: {type(e).__name__}: {e}")

# Check what Python sees when it tries to import
print("\n--- Import Path Resolution ---")
try:
    import metta

    print(f"metta module: {metta.__file__}")
    print(f"metta path: {metta.__path__}")

    import metta.rl

    print(f"metta.rl module: {metta.rl.__file__}")
except Exception as e:
    print(f"Error importing parent modules: {e}")

print("\n=== End Diagnostics ===")
