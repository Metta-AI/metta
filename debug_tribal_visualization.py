#!/usr/bin/env python3
"""
Debug script for tribal visualization SIGSEGV issues.
Tests different approaches to isolate the crash source.
"""

import sys
import os
import subprocess
import time
from pathlib import Path

def test_basic_bindings():
    """Test 1: Verify basic bindings work without GUI"""
    print("=" * 60)
    print("🧪 TEST 1: Basic Bindings (No GUI)")
    print("=" * 60)
    
    tribal_dir = Path(__file__).parent / "tribal"
    test_script = tribal_dir / "test_tribal_bindings.py"
    
    if not test_script.exists():
        print(f"❌ Test script not found: {test_script}")
        return False
    
    try:
        result = subprocess.run(
            [sys.executable, str(test_script)],
            cwd=str(tribal_dir),
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            print("✅ Basic bindings work without GUI")
            print("   Environment creation, stepping, and text rendering successful")
            return True
        else:
            print("❌ Basic bindings failed:")
            print(f"   stdout: {result.stdout}")
            print(f"   stderr: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏱️  Basic bindings test timed out")
        return False
    except Exception as e:
        print(f"❌ Error running basic bindings test: {e}")
        return False

def test_nimpy_import():
    """Test 2: Can we import the nimpy viewer module?"""
    print("\n" + "=" * 60)
    print("🧪 TEST 2: Nimpy Viewer Import")
    print("=" * 60)
    
    tribal_dir = Path(__file__).parent / "tribal"
    old_cwd = os.getcwd()
    
    try:
        os.chdir(tribal_dir)
        sys.path.insert(0, str(tribal_dir))
        
        print("🔄 Attempting to import tribal_nimpy_viewer...")
        import tribal_nimpy_viewer as viewer
        print("✅ Successfully imported tribal_nimpy_viewer")
        
        # Test if we can call the functions without crashing
        print("🔧 Testing initVisualization() call...")
        # Note: This might crash, so we're testing it
        result = viewer.initVisualization()
        print(f"✅ initVisualization() returned: {result}")
        
        if result:
            print("🧹 Cleaning up...")
            viewer.closeVisualization()
            print("✅ Cleanup successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Nimpy viewer import/init failed: {e}")
        return False
    finally:
        os.chdir(old_cwd)
        if str(tribal_dir) in sys.path:
            sys.path.remove(str(tribal_dir))

def test_subprocess_approach():
    """Test 3: Native Nim viewer as subprocess"""
    print("\n" + "=" * 60)
    print("🧪 TEST 3: Native Nim Subprocess")
    print("=" * 60)
    
    tribal_dir = Path(__file__).parent / "tribal"
    
    try:
        print("🚀 Launching native Nim viewer as subprocess...")
        
        # Try to compile and run the native Nim viewer
        compile_cmd = ["nim", "c", "-r", "-d:release", "src/tribal"]
        
        print(f"🔨 Running: {' '.join(compile_cmd)}")
        process = subprocess.Popen(
            compile_cmd,
            cwd=str(tribal_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Let it run for a few seconds
        time.sleep(3)
        
        # Check if still running
        if process.poll() is None:
            print("✅ Native Nim viewer started successfully")
            print("🔥 Terminating after 3 seconds...")
            process.terminate()
            process.wait(timeout=5)
            return True
        else:
            stdout, stderr = process.communicate()
            print(f"❌ Native Nim viewer exited with code {process.returncode}")
            print(f"   stdout: {stdout}")
            print(f"   stderr: {stderr}")
            return False
            
    except FileNotFoundError:
        print("❌ Nim compiler not found. Install with: brew install nim")
        return False
    except Exception as e:
        print(f"❌ Error running subprocess test: {e}")
        return False

def test_genny_only():
    """Test 4: Use genny bindings without any visualization"""
    print("\n" + "=" * 60)
    print("🧪 TEST 4: Genny Bindings Only (No Visualization)")
    print("=" * 60)
    
    try:
        # Add bindings to path
        tribal_dir = Path(__file__).parent / "tribal"
        bindings_path = tribal_dir / "bindings" / "generated"
        
        if not bindings_path.exists():
            print("🔨 Building bindings first...")
            build_result = subprocess.run(
                ["bash", "build_bindings.sh"],
                cwd=str(tribal_dir),
                capture_output=True,
                text=True
            )
            
            if build_result.returncode != 0:
                print(f"❌ Failed to build bindings: {build_result.stderr}")
                return False
        
        sys.path.insert(0, str(bindings_path))
        
        print("🔄 Importing tribal genny bindings...")
        import tribal
        
        print("✅ Genny bindings imported successfully")
        
        # Test basic operations
        print("🔧 Testing basic environment operations...")
        config = tribal.default_tribal_config()
        env = tribal.new_tribal_env(config)
        
        print("✅ Environment created")
        
        # Test stepping without visualization
        env.reset_env()
        actions = [0, 0] * 15  # NOOP actions for all 15 agents
        success = env.step(actions)
        
        if success:
            print("✅ Environment stepping works")
            
            # Test text rendering (should work)
            text = env.render_text()
            print(f"✅ Text rendering works (length: {len(text)})")
            
            return True
        else:
            print("❌ Environment stepping failed")
            return False
            
    except Exception as e:
        print(f"❌ Genny-only test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all debugging tests"""
    print("🔬 Tribal Visualization Debug Suite")
    print("🎯 Goal: Isolate SIGSEGV crash source")
    print("📋 Running 4 progressive tests...")
    
    tests = [
        ("Basic Bindings (No GUI)", test_basic_bindings),
        ("Nimpy Viewer Import", test_nimpy_import),
        ("Native Nim Subprocess", test_subprocess_approach),
        ("Genny Bindings Only", test_genny_only),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
            
            if result:
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
                
        except Exception as e:
            print(f"💥 {test_name}: CRASHED - {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 DEBUG RESULTS SUMMARY")
    print("=" * 60)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {test_name}")
    
    # Recommendations
    print("\n🎯 RECOMMENDATIONS:")
    
    if results[0][1]:  # Basic bindings work
        print("✅ Basic Python-Nim communication works")
        
        if not results[1][1]:  # Nimpy viewer fails
            print("💡 Issue is in nimpy OpenGL visualization")
            print("   Recommend: Use subprocess approach (Test 3)")
            
        if results[2][1]:  # Subprocess works
            print("✅ Native Nim viewer works as subprocess")
            print("   Solution: Use separate process architecture")
            
        if results[3][1]:  # Genny works
            print("✅ Genny bindings work without visualization")
            print("   Alternative: Headless mode for training")
    else:
        print("❌ Basic bindings broken - fix build first")
    
    print("\n🔧 NEXT STEPS:")
    print("1. If Test 3 passed: Use subprocess architecture")
    print("2. If only Test 4 passed: Run headless for now")
    print("3. If all failed: Check Nim installation and dependencies")

if __name__ == "__main__":
    main()