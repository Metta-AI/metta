#!/usr/bin/env python3
"""
Isolate the exact crash point in nimpy rendering.
We know init works, so test each rendering operation separately.
"""

import os
import sys
from pathlib import Path

def test_minimal_rendering_sequence():
    """Test each step of the rendering sequence to find the crash point"""
    print("🔬 Testing Nimpy Rendering Sequence Step by Step")
    print("=" * 60)
    
    try:
        # Setup
        tribal_dir = Path(__file__).parent / "tribal"
        old_cwd = os.getcwd()
        os.chdir(tribal_dir)
        
        # Add to path
        sys.path.insert(0, str(tribal_dir))
        
        # Import viewer
        print("1️⃣  Importing nimpy viewer...")
        import tribal_nimpy_viewer as viewer
        print("✅ Import successful")
        
        # Test initialization (we know this works)
        print("2️⃣  Testing initialization...")
        if not viewer.initVisualization():
            print("❌ Initialization failed")
            return False
        print("✅ Initialization successful")
        
        # Test window status check
        print("3️⃣  Testing window status...")
        if not viewer.isWindowOpen():
            print("❌ Window not open")
            return False
        print("✅ Window is open")
        
        # Load assets (we know this works in native)
        print("4️⃣  Testing asset loading...")
        if not viewer.loadAssets():
            print("⚠️  Asset loading failed, continuing without assets")
        else:
            print("✅ Assets loaded successfully")
        
        # NOW TEST THE CRITICAL RENDERING CALL
        print("5️⃣  🚨 CRITICAL TEST: First renderFrame() call...")
        print("    This is where the crash likely happens...")
        
        try:
            # This should crash with SIGSEGV
            result = viewer.renderFrame()
            print(f"✅ UNEXPECTED SUCCESS: renderFrame() returned: {result}")
            
            # If we got here, try a few more frames
            print("6️⃣  Testing additional render frames...")
            for i in range(3):
                result = viewer.renderFrame()
                print(f"    Frame {i+2}: {result}")
                if not result:
                    print(f"❌ Rendering failed at frame {i+2}")
                    break
            
        except Exception as render_error:
            print(f"💥 CRASH CONFIRMED: renderFrame() crashed: {render_error}")
            import traceback
            traceback.print_exc()
        
        # Cleanup
        print("7️⃣  Cleanup...")
        viewer.closeVisualization()
        print("✅ Cleanup complete")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        os.chdir(old_cwd)

def test_alternative_rendering_approach():
    """Test if we can render without the full rendering pipeline"""
    print("\n🔬 Testing Alternative Rendering Approach")
    print("=" * 60)
    
    # Try to create a custom nimpy module that does minimal OpenGL operations
    minimal_code = '''
import nimpy
import boxy, opengl, windy, vmath

var window: Window
var bxy: Boxy

proc initMinimal*(): bool {.exportpy.} =
  try:
    window = newWindow("Minimal Test", ivec2(400, 300))
    makeContextCurrent(window)
    when not defined(emscripten):
      loadExtensions()
    bxy = newBoxy()
    return true
  except:
    return false

proc renderMinimal*(): bool {.exportpy.} =
  try:
    # Try the absolute minimum rendering operations
    bxy.beginFrame(window.size)
    # Skip all drawing
    bxy.endFrame()
    window.swapBuffers()
    return true
  except:
    return false

proc isOpenMinimal*(): bool {.exportpy.} =
  return not window.closeRequested

proc closeMinimal*() {.exportpy.} =
  discard
'''
    
    tribal_dir = Path(__file__).parent / "tribal"
    minimal_file = tribal_dir / "src" / "minimal_render_test.nim"
    
    try:
        # Write minimal test file
        with open(minimal_file, 'w') as f:
            f.write(minimal_code)
        
        print("📝 Created minimal render test")
        
        # Compile it
        import subprocess
        result = subprocess.run([
            "nim", "c", "--app:lib", "--out:minimal_render_test.so",
            "-d:release", "src/minimal_render_test.nim"
        ], cwd=str(tribal_dir), capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Compilation failed: {result.stderr}")
            return False
        
        print("✅ Compiled minimal test")
        
        # Test it
        old_cwd = os.getcwd()
        os.chdir(tribal_dir)
        
        try:
            sys.path.insert(0, str(tribal_dir))
            import minimal_render_test as test
            
            print("🔧 Testing minimal initialization...")
            if test.initMinimal():
                print("✅ Minimal init works")
                
                print("🔧 Testing minimal rendering...")
                for i in range(5):
                    if test.renderMinimal():
                        print(f"✅ Minimal render {i+1} works")
                    else:
                        print(f"❌ Minimal render {i+1} failed")
                        break
                        
                    if not test.isOpenMinimal():
                        print("🪟 Window closed")
                        break
                
                test.closeMinimal()
                print("✅ Minimal rendering test completed successfully!")
                return True
            else:
                print("❌ Minimal init failed")
                
        finally:
            os.chdir(old_cwd)
            # Cleanup
            minimal_file.unlink(missing_ok=True)
            (tribal_dir / "minimal_render_test.so").unlink(missing_ok=True)
        
    except Exception as e:
        print(f"❌ Alternative rendering test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all rendering crash tests"""
    print("🎯 Goal: Find the exact crash point in nimpy rendering")
    
    # Test 1: Step-by-step rendering to find crash
    success1 = test_minimal_rendering_sequence()
    
    # Test 2: Alternative minimal rendering
    success2 = test_alternative_rendering_approach()
    
    print(f"\n📊 RESULTS:")
    print(f"   Step-by-step test: {'✅ SUCCESS' if success1 else '❌ CRASH'}")
    print(f"   Minimal render test: {'✅ SUCCESS' if success2 else '❌ FAIL'}")
    
    if success2 and not success1:
        print(f"\n💡 SOLUTION FOUND:")
        print(f"   The crash is in the complex rendering pipeline, not OpenGL itself")
        print(f"   Use minimal rendering approach for Python ↔ Nim ↔ GUI communication")
    elif success1:
        print(f"\n❓ UNEXPECTED:")
        print(f"   The crash may be intermittent or environment-dependent")
    else:
        print(f"\n🚨 FUNDAMENTAL ISSUE:")
        print(f"   OpenGL rendering incompatible with nimpy in this environment")

if __name__ == "__main__":
    main()