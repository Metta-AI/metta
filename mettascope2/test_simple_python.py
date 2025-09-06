#!/usr/bin/env python3
"""Test the simple library to verify export mechanism works"""
import ctypes
import os

# Load simple test library
lib_path = os.path.join(os.path.dirname(__file__), "test_simple.dylib")
print(f"Loading library from: {lib_path}")

try:
    lib = ctypes.cdll.LoadLibrary(lib_path)
    print("✓ Library loaded successfully")
    
    # Test the simple function
    try:
        test_func = lib.test_function
        test_func.restype = ctypes.c_int
        result = test_func()
        print(f"✓ test_function() returned: {result}")
        
    except AttributeError as e:
        print(f"✗ Function not found: {e}")
        
except OSError as e:
    print(f"✗ Failed to load library: {e}")

# Instructions for building the test library
print("\nTo build the test library:")
print("nim c --app:lib --mm:arc --opt:speed --out:test_simple.dylib test_simple.nim")