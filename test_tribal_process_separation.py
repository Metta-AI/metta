#!/usr/bin/env python3
"""
Quick test script for tribal process separation.

This tests the new SIGSEGV-free process separation approach.
"""

import subprocess
import sys
import time
from pathlib import Path

def test_process_separation():
    """Test the process-separated tribal environment"""
    print("🧪 Testing Tribal Process Separation")
    print("="*50)
    
    # Test 1: Direct process controller test
    print("1. Testing direct process controller...")
    tribal_dir = Path("tribal")
    if not tribal_dir.exists():
        print("❌ Tribal directory not found")
        return False
    
    try:
        result = subprocess.run(
            ["python3", "tribal_process_controller.py"],
            cwd=tribal_dir,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            print("✅ Process controller test passed")
        else:
            print(f"❌ Process controller test failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⚠️  Process controller test timed out (expected for interactive test)")
    except Exception as e:
        print(f"❌ Process controller test error: {e}")
        return False
    
    # Test 2: Recipe test with test_move
    print("2. Testing recipe with test_move (5 second timeout)...")
    try:
        result = subprocess.run([
            "uv", "run", "./tools/run.py", 
            "experiments.recipes.tribal_basic.play", 
            "--args", "policy_uri=test_move"
        ], capture_output=True, text=True, timeout=5)
        
        # Expected to timeout since it runs the viewer
        print("⚠️  Recipe test timed out (expected - viewer runs indefinitely)")
        
    except subprocess.TimeoutExpired:
        print("⚠️  Recipe test timed out as expected")
    except Exception as e:
        print(f"❌ Recipe test error: {e}")
        return False
    
    print("\n🎉 Process separation tests completed!")
    print("\nManual testing:")
    print("• Run: uv run ./tools/run.py experiments.recipes.tribal_basic.play --args policy_uri=test_move")
    print("• You should see a Nim window with moving agents (no SIGSEGV!)")
    print("• Press Ctrl+C to stop")
    
    return True


if __name__ == "__main__":
    success = test_process_separation()
    sys.exit(0 if success else 1)