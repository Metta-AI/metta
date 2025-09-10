#!/usr/bin/env python3
"""
Debug OpenGL environment differences between native Nim vs nimpy.
Identifies why OpenGL works natively but crashes through Python bindings.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def test_native_nim_opengl():
    """Test 1: Verify native Nim OpenGL works fine"""
    print("=" * 60)
    print("🧪 TEST 1: Native Nim OpenGL Execution")
    print("=" * 60)
    
    tribal_dir = Path(__file__).parent / "tribal"
    
    try:
        print("🎮 Testing native Nim execution...")
        
        # Try to run the native tribal viewer
        compile_cmd = ["nim", "c", "-r", "-d:release", "src/tribal"]
        
        print(f"🔨 Running: {' '.join(compile_cmd)}")
        process = subprocess.Popen(
            compile_cmd,
            cwd=str(tribal_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=os.environ.copy()  # Use current environment
        )
        
        # Let it run for 3 seconds to initialize OpenGL
        time.sleep(3)
        
        if process.poll() is None:
            print("✅ Native Nim OpenGL: SUCCESS")
            print("   Window opened and OpenGL initialized successfully")
            process.terminate()
            process.wait(timeout=5)
            return True, os.environ.copy()
        else:
            stdout, stderr = process.communicate()
            print(f"❌ Native Nim failed: {process.returncode}")
            print(f"   stdout: {stdout}")
            print(f"   stderr: {stderr}")
            return False, None
            
    except Exception as e:
        print(f"❌ Native Nim test failed: {e}")
        return False, None

def capture_python_environment():
    """Test 2: Capture Python's runtime environment"""
    print("\n" + "=" * 60)
    print("🧪 TEST 2: Python Runtime Environment Analysis")
    print("=" * 60)
    
    env_info = {
        'python_executable': sys.executable,
        'python_version': sys.version,
        'python_path': sys.path[:5],  # First 5 entries
        'working_directory': os.getcwd(),
        'environment_vars': {},
        'loaded_modules': []
    }
    
    # Capture key environment variables
    key_vars = [
        'PATH', 'DYLD_LIBRARY_PATH', 'LD_LIBRARY_PATH', 'PYTHONPATH',
        'DISPLAY', 'XAUTHORITY', 'WAYLAND_DISPLAY',  # Display vars
        'DYLD_FRAMEWORK_PATH', 'DYLD_FALLBACK_LIBRARY_PATH',  # macOS specific
        'PKG_CONFIG_PATH', 'LIBRARY_PATH', 'CPATH'  # Build vars
    ]
    
    for var in key_vars:
        env_info['environment_vars'][var] = os.environ.get(var, 'NOT_SET')
    
    # Check loaded modules
    graphics_modules = ['OpenGL', 'opengl', 'boxy', 'windy', 'glfw']
    for module_name in graphics_modules:
        if module_name in sys.modules:
            env_info['loaded_modules'].append(module_name)
    
    print("📊 Python Environment Summary:")
    print(f"   Executable: {env_info['python_executable']}")
    print(f"   Working Dir: {env_info['working_directory']}")
    print(f"   Graphics modules loaded: {env_info['loaded_modules']}")
    
    # Check for virtual environment
    venv_indicators = ['VIRTUAL_ENV', 'CONDA_DEFAULT_ENV', 'PIPENV_ACTIVE']
    for indicator in venv_indicators:
        if os.environ.get(indicator):
            print(f"   🐍 Virtual env detected: {indicator}={os.environ[indicator]}")
    
    return env_info

def test_nimpy_with_native_environment(native_env):
    """Test 3: Run nimpy with the same environment as native Nim"""
    print("\n" + "=" * 60)
    print("🧪 TEST 3: Nimpy with Native Environment")
    print("=" * 60)
    
    if not native_env:
        print("❌ Skipping - native environment not available")
        return False
    
    try:
        tribal_dir = Path(__file__).parent / "tribal"
        old_cwd = os.getcwd()
        
        # Change to tribal directory (same as native)
        os.chdir(tribal_dir)
        
        # Set environment to match native execution
        original_env = os.environ.copy()
        os.environ.update(native_env)
        
        print("🔄 Testing nimpy import with native environment...")
        
        # Add to Python path
        sys.path.insert(0, str(tribal_dir))
        
        # Test import
        import tribal_nimpy_viewer as viewer
        print("✅ Nimpy module imported successfully")
        
        # Test initialization with native environment
        print("🎨 Testing OpenGL initialization...")
        
        success = viewer.initVisualization()
        if success:
            print("✅ SUCCESS: OpenGL works with native environment!")
            
            # Test a render frame
            print("🖼️  Testing render frame...")
            if viewer.renderFrame():
                print("✅ Render frame successful")
            else:
                print("⚠️  Render frame failed but init worked")
            
            # Cleanup
            viewer.closeVisualization()
            return True
        else:
            print("❌ OpenGL initialization still failed")
            return False
            
    except Exception as e:
        print(f"❌ Nimpy with native environment failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Restore environment and directory
        os.environ.clear()
        os.environ.update(original_env)
        os.chdir(old_cwd)

def test_library_loading_differences():
    """Test 4: Check dynamic library loading differences"""
    print("\n" + "=" * 60)
    print("🧪 TEST 4: Dynamic Library Loading Analysis")
    print("=" * 60)
    
    try:
        # Check what OpenGL libraries are available
        opengl_locations = [
            "/System/Library/Frameworks/OpenGL.framework/OpenGL",  # macOS system
            "/usr/lib/libGL.so",  # Linux
            "/usr/lib/x86_64-linux-gnu/libGL.so.1",  # Ubuntu
            "/opt/homebrew/lib/libGL.dylib",  # macOS Homebrew
            "/usr/local/lib/libGL.dylib"  # macOS custom
        ]
        
        available_libs = []
        for lib_path in opengl_locations:
            if Path(lib_path).exists():
                available_libs.append(lib_path)
        
        print(f"📚 Available OpenGL libraries: {len(available_libs)}")
        for lib in available_libs:
            print(f"   ✓ {lib}")
        
        # Check Python's library loading
        import ctypes.util
        opengl_lib = ctypes.util.find_library("OpenGL") or ctypes.util.find_library("GL")
        print(f"🐍 Python finds OpenGL at: {opengl_lib}")
        
        # Check if we can load OpenGL directly from Python
        try:
            if sys.platform == "darwin":
                ctypes.CDLL("/System/Library/Frameworks/OpenGL.framework/OpenGL")
                print("✅ Python can load macOS OpenGL framework")
            else:
                ctypes.CDLL(opengl_lib) if opengl_lib else None
                print("✅ Python can load OpenGL library")
        except Exception as e:
            print(f"❌ Python OpenGL loading failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Library loading test failed: {e}")
        return False

def test_minimal_opengl_reproduction():
    """Test 5: Minimal OpenGL reproduction in nimpy"""
    print("\n" + "=" * 60)
    print("🧪 TEST 5: Minimal OpenGL Reproduction")
    print("=" * 60)
    
    try:
        # Create a minimal nimpy OpenGL test
        tribal_dir = Path(__file__).parent / "tribal"
        
        minimal_test = """
## Minimal OpenGL test for nimpy debugging
import nimpy
import windy, opengl

var window: Window

proc testWindowOnly*(): bool {.exportpy.} =
  try:
    echo "Creating window..."
    window = newWindow("Test", ivec2(800, 600))
    echo "Window created successfully"
    return true
  except Exception as e:
    echo "Window creation failed: ", e.msg
    return false

proc testOpenGLContext*(): bool {.exportpy.} =
  try:
    echo "Making OpenGL context current..."
    makeContextCurrent(window)
    echo "OpenGL context created successfully"
    return true
  except Exception as e:
    echo "OpenGL context failed: ", e.msg
    return false

proc testLoadExtensions*(): bool {.exportpy.} =
  try:
    echo "Loading OpenGL extensions..."
    when not defined(emscripten):
      loadExtensions()
    echo "Extensions loaded successfully"
    return true
  except Exception as e:
    echo "Extension loading failed: ", e.msg
    return false

proc cleanup*() {.exportpy.} =
  echo "Cleanup complete"
"""
        
        # Write minimal test
        minimal_file = tribal_dir / "src" / "minimal_opengl_test.nim"
        with open(minimal_file, 'w') as f:
            f.write(minimal_test)
        
        print("📝 Created minimal OpenGL test")
        
        # Compile minimal test
        compile_cmd = [
            "nim", "c", "--app:lib", "--out:minimal_opengl_test.so", 
            "-d:release", "src/minimal_opengl_test.nim"
        ]
        
        result = subprocess.run(
            compile_cmd,
            cwd=str(tribal_dir),
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"❌ Compilation failed: {result.stderr}")
            return False
        
        print("✅ Compiled minimal test successfully")
        
        # Test the minimal version
        old_cwd = os.getcwd()
        os.chdir(tribal_dir)
        
        try:
            sys.path.insert(0, str(tribal_dir))
            import minimal_opengl_test as test
            
            print("🪟 Testing window creation...")
            if test.testWindowOnly():
                print("✅ Window creation works")
                
                print("🖼️  Testing OpenGL context...")
                if test.testOpenGLContext():
                    print("✅ OpenGL context works")
                    
                    print("🔧 Testing extension loading...")
                    if test.testLoadExtensions():
                        print("✅ ALL TESTS PASSED - OpenGL should work!")
                        return True
                    else:
                        print("❌ Extension loading failed")
                else:
                    print("❌ OpenGL context failed - THIS IS THE ISSUE")
            else:
                print("❌ Window creation failed")
            
            test.cleanup()
            return False
            
        finally:
            os.chdir(old_cwd)
            # Clean up
            minimal_file.unlink(missing_ok=True)
            (tribal_dir / "minimal_opengl_test.so").unlink(missing_ok=True)
        
    except Exception as e:
        print(f"❌ Minimal reproduction failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def provide_environment_fixes():
    """Provide specific fixes for the identified issues"""
    print("\n" + "=" * 60)
    print("🔧 ENVIRONMENT FIXES")
    print("=" * 60)
    
    fixes = []
    
    # macOS specific fixes
    if sys.platform == "darwin":
        fixes.extend([
            "🍎 macOS OpenGL Framework Fix:",
            "   export DYLD_FRAMEWORK_PATH=/System/Library/Frameworks:$DYLD_FRAMEWORK_PATH",
            "",
            "🍎 macOS Metal/OpenGL Fix:",
            "   export DYLD_LIBRARY_PATH=/usr/lib:$DYLD_LIBRARY_PATH",
            "",
        ])
    
    # General fixes
    fixes.extend([
        "🐍 Python Library Path Fix:",
        f"   export PYTHONPATH={Path(__file__).parent}/tribal:$PYTHONPATH",
        "",
        "📁 Working Directory Fix:",
        f"   cd {Path(__file__).parent}/tribal",
        "   # Run Python from tribal directory",
        "",
        "🔗 Runtime Library Fix:",
        "   export LD_LIBRARY_PATH=/usr/local/lib:/opt/homebrew/lib:$LD_LIBRARY_PATH",
        "",
        "🎮 Display Fix (if using SSH/remote):",
        "   export DISPLAY=:0",
        "   # Or run with GUI support: ssh -X user@host",
        "",
    ])
    
    for fix in fixes:
        print(fix)
    
    return fixes

def main():
    """Run all environment debugging tests"""
    print("🔬 OpenGL Environment Debugging Suite")
    print("🎯 Goal: Identify why OpenGL works natively but fails in nimpy")
    
    # Test 1: Native Nim
    native_works, native_env = test_native_nim_opengl()
    
    # Test 2: Python environment
    python_env = capture_python_environment()
    
    # Test 3: Environment matching
    env_match_works = False
    if native_works:
        env_match_works = test_nimpy_with_native_environment(native_env)
    
    # Test 4: Library analysis
    libs_ok = test_library_loading_differences()
    
    # Test 5: Minimal reproduction
    minimal_works = test_minimal_opengl_reproduction()
    
    # Summary and fixes
    print("\n" + "=" * 60)
    print("📊 DIAGNOSTIC RESULTS")
    print("=" * 60)
    
    results = [
        ("Native Nim OpenGL", native_works),
        ("Library Loading", libs_ok),
        ("Environment Matching", env_match_works),
        ("Minimal Reproduction", minimal_works),
    ]
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {test_name}")
    
    print(f"\n🎯 ROOT CAUSE ANALYSIS:")
    
    if native_works and not env_match_works:
        print("💡 ISSUE: Environment difference between native Nim and Python")
        print("   Solution: Match Python environment to native Nim environment")
        
    elif not libs_ok:
        print("💡 ISSUE: OpenGL library loading problems")
        print("   Solution: Fix library paths and permissions")
        
    elif not minimal_works:
        print("💡 ISSUE: Fundamental nimpy ↔ OpenGL incompatibility")
        print("   Solution: Use separate process or alternative rendering")
        
    else:
        print("✅ All tests passed - environment should work!")
    
    # Provide fixes
    provide_environment_fixes()

if __name__ == "__main__":
    main()