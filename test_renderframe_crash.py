#!/usr/bin/env python3
"""
Test to confirm renderFrame() is where the SIGSEGV crash happens
"""

import sys
import os
from pathlib import Path

def test_renderframe_crash():
    """Isolate the renderFrame crash"""
    print("🎯 Testing renderFrame() crash isolation")
    
    try:
        # Setup
        tribal_dir = Path(__file__).parent / "tribal"
        old_cwd = os.getcwd()
        os.chdir(tribal_dir)
        sys.path.insert(0, str(tribal_dir))
        
        # Import and setup
        import tribal_nimpy_viewer as viewer
        print("✅ 1. Import successful")
        
        # Initialize (we know this works)
        if not viewer.initVisualization():
            print("❌ 2. Initialization failed")
            return
        print("✅ 2. Initialization successful")
        
        # Load assets (we know this works too)
        if not viewer.loadAssets():
            print("❌ 3. Asset loading failed")
            return
        print("✅ 3. Asset loading successful")
        
        print("🚨 4. About to call renderFrame() - this should SIGSEGV crash...")
        
        # This should crash
        result = viewer.renderFrame()
        
        # If we get here, it didn't crash
        print(f"🤔 4. Unexpected success: renderFrame() returned {result}")
        
    except Exception as e:
        print(f"💥 4. CRASHED as expected: {e}")
    finally:
        os.chdir(old_cwd)
        print("✅ Test complete")

if __name__ == "__main__":
    test_renderframe_crash()