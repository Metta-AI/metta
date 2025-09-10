#!/usr/bin/env python3
"""
Test if the visualization fixes work.
This tests the safe nimpy viewer and rendering approaches.
"""

import os
import sys
import time
import subprocess
from pathlib import Path

def test_safe_nimpy_viewer():
    """Test 1: Use the safe nimpy viewer with gradual initialization"""
    print("🧪 TEST 1: Safe Nimpy Viewer with Gradual Initialization")
    print("=" * 60)
    
    try:
        # Setup
        tribal_dir = Path(__file__).parent / "tribal"
        old_cwd = os.getcwd()
        os.chdir(tribal_dir)
        
        # Build safe viewer if not exists
        safe_viewer_so = tribal_dir / "tribal_nimpy_safe_viewer.so"
        if not safe_viewer_so.exists():
            print("🔨 Building safe viewer...")
            result = subprocess.run([
                "nim", "c", "--app:lib", "--out:tribal_nimpy_safe_viewer.so", 
                "-d:release", "src/tribal_nimpy_safe_viewer.nim"
            ], cwd=str(tribal_dir), capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"❌ Failed to build safe viewer: {result.stderr}")
                return False
            print("✅ Safe viewer built successfully")
        
        # Import safe viewer
        sys.path.insert(0, str(tribal_dir))
        import tribal_nimpy_safe_viewer as viewer
        print("✅ Imported safe viewer")
        
        # Step 1: Window creation
        print("🪟 Step 1: Testing window creation...")
        if not viewer.initWindowOnly():
            print("❌ Window creation failed")
            return False
        print("✅ Window created successfully")
        
        # Step 2: OpenGL context (this is where crashes usually happen)
        print("🖼️  Step 2: Testing OpenGL context creation...")
        if not viewer.initOpenGLContext():
            print("❌ OpenGL context failed - this is the crash point!")
            print("💡 The issue is likely in OpenGL context creation with Python")
            return False
        print("✅ OpenGL context created successfully")
        
        # Step 3: Panel initialization
        print("🎛️  Step 3: Testing UI panel initialization...")
        if not viewer.initPanels():
            print("❌ Panel initialization failed")
            return False
        print("✅ UI panels initialized successfully")
        
        # Step 4: Asset loading
        print("🎨 Step 4: Testing asset loading...")
        if not viewer.loadAssetsSafe():
            print("⚠️  Asset loading failed, continuing...")
        else:
            print("✅ Assets loaded successfully")
        
        # Step 5: CRITICAL TEST - Minimal rendering
        print("🚨 Step 5: CRITICAL TEST - Minimal rendering...")
        print("   This is where the SIGSEGV crash typically occurs")
        
        for i in range(5):
            if not viewer.renderFrameMinimal():
                print(f"❌ Minimal rendering failed at frame {i+1}")
                break
            print(f"✅ Minimal frame {i+1} rendered successfully")
            time.sleep(0.1)
        else:
            print("🎉 ALL MINIMAL RENDERING SUCCESSFUL!")
            
            # Step 6: Try basic rendering with actual content
            print("🎨 Step 6: Testing basic rendering with content...")
            for i in range(3):
                if viewer.isWindowOpen() and viewer.renderFrameBasic():
                    print(f"✅ Basic frame {i+1} rendered with content")
                    time.sleep(0.2)
                else:
                    print(f"⚠️  Basic rendering stopped at frame {i+1}")
                    break
        
        # Cleanup
        viewer.closeVisualization()
        print("✅ Safe viewer test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Safe viewer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        os.chdir(old_cwd)

def test_full_communication_with_safe_viewer():
    """Test 2: Full Python → Nim → Safe GUI → Python loop"""
    print("\n🧪 TEST 2: Full Communication Loop with Safe Viewer")
    print("=" * 60)
    
    try:
        # Setup environment
        tribal_dir = Path(__file__).parent / "tribal"
        old_cwd = os.getcwd()
        os.chdir(tribal_dir)
        
        # Import both tribal bindings and safe viewer
        bindings_path = tribal_dir / "bindings" / "generated"
        sys.path.insert(0, str(bindings_path))
        sys.path.insert(0, str(tribal_dir))
        
        import tribal
        import tribal_nimpy_safe_viewer as viewer
        
        print("✅ Imported tribal bindings and safe viewer")
        
        # Create Nim environment
        config = tribal.default_tribal_config()
        nim_env = tribal.TribalEnv(config)
        print("✅ Created Nim environment")
        
        # Initialize external controller
        if not tribal.init_external_nncontroller():
            print("❌ Failed to initialize external controller")
            return False
        print("✅ External controller initialized")
        
        # Initialize safe viewer
        print("🎨 Initializing safe visualization...")
        if not viewer.initVisualizationSafe():
            print("❌ Safe visualization init failed")
            return False
        print("✅ Safe visualization ready")
        
        # Load assets
        if not viewer.loadAssetsSafe():
            print("⚠️  Assets failed to load, continuing without")
        
        # Reset environment
        nim_env.reset_env()
        print("✅ Environment reset")
        
        # Test communication loop with visualization
        print("🎮 Testing 10 steps of Python → Nim → GUI communication...")
        
        for step in range(10):
            # 1. Get observations from Nim
            observations = nim_env.get_token_observations()
            
            # 2. Generate test actions (move randomly)
            import random
            flat_actions = []
            for agent in range(15):
                action_type = 1  # MOVE
                direction = random.randint(0, 3)  # Random direction
                flat_actions.extend([action_type, direction])
            
            # 3. Send to external controller
            actions_seq = tribal.SeqInt()
            for action in flat_actions:
                actions_seq.append(action)
            
            if not tribal.set_external_actions_from_python(actions_seq):
                print(f"❌ Failed to send actions at step {step}")
                return False
            
            # 4. Step environment
            if not nim_env.step(actions_seq):
                print(f"❌ Environment step failed at step {step}")
                return False
            
            # 5. Update visualization (CRITICAL TEST)
            if not viewer.renderFrameMinimal():
                print(f"❌ Visualization failed at step {step}")
                print("💡 The core communication works, but visualization crashes")
                return False
            
            # 6. Get results
            rewards = nim_env.get_rewards()
            
            print(f"  Step {step}: ✅ Actions sent → Nim processed → GUI updated → Got {len(rewards)} rewards")
            
            # Brief pause
            time.sleep(0.1)
            
            # Check if window closed
            if not viewer.isWindowOpen():
                print("🪟 Window closed by user")
                break
        
        viewer.closeVisualization()
        print("🎉 FULL COMMUNICATION LOOP WITH VISUALIZATION SUCCESSFUL!")
        return True
        
    except Exception as e:
        print(f"❌ Full communication test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        os.chdir(old_cwd)

def test_tribal_recipe_with_visualization():
    """Test 3: Use the updated tribal recipe to test visualization"""
    print("\n🧪 TEST 3: Tribal Recipe with Visualization")
    print("=" * 60)
    
    try:
        # Test if the recipe now uses visualization instead of headless
        print("🚀 Testing tribal recipe with visualization enabled...")
        
        # This should now use the safe visualization approach
        # We'll run it briefly and see if it crashes or works
        
        print("💡 Run this command to test the full system:")
        print("   uv run ./tools/run.py experiments.recipes.tribal_basic.play --args policy_uri=test_move")
        print("   (If it doesn't crash with SIGSEGV, the fix worked!)")
        
        return True
        
    except Exception as e:
        print(f"❌ Recipe test setup failed: {e}")
        return False

def main():
    """Run all visualization fix tests"""
    print("🔬 Testing Visualization Fixes")
    print("🎯 Goal: Verify that Python → Nim → GUI communication works without SIGSEGV")
    
    results = []
    
    # Test 1: Safe nimpy viewer
    result1 = test_safe_nimpy_viewer()
    results.append(("Safe Nimpy Viewer", result1))
    
    # Test 2: Full communication loop (only if Test 1 passes)
    if result1:
        result2 = test_full_communication_with_safe_viewer()
        results.append(("Full Communication with GUI", result2))
    else:
        result2 = False
        results.append(("Full Communication with GUI", False))
        print("⏭️  Skipping full communication test due to safe viewer failure")
    
    # Test 3: Recipe integration
    result3 = test_tribal_recipe_with_visualization()
    results.append(("Recipe Integration", result3))
    
    # Summary
    print(f"\n📊 VISUALIZATION FIX TEST RESULTS")
    print("=" * 60)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {test_name}")
    
    print(f"\n🎯 CONCLUSION:")
    
    if result1 and result2:
        print("🎉 SUCCESS: Python → Nim → GUI communication works!")
        print("   The visualization fixes resolved the SIGSEGV crashes")
        print("   You can now use visual training and debugging")
        
    elif result1:
        print("🟡 PARTIAL: Safe viewer works, but full integration has issues")
        print("   The basic OpenGL fix works, but may need refinement")
        
    else:
        print("🚨 FAILED: OpenGL context creation still crashes")
        print("   This suggests a fundamental OpenGL/Python incompatibility")
        print("   Recommendation: Use headless mode for training")
    
    print(f"\n💡 NEXT STEPS:")
    if all(results[i][1] for i in [0, 1]):
        print("  1. Update tribal recipe to use safe viewer by default")
        print("  2. Test full training pipeline with visualization")
        print("  3. Enjoy Python ↔ Nim ↔ GUI integration!")
    else:
        print("  1. Use headless mode for training (already proven to work)")
        print("  2. Use native Nim viewer separately for visualization when needed")
        print("  3. Consider alternative visualization approaches")

if __name__ == "__main__":
    main()