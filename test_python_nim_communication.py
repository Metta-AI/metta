#!/usr/bin/env python3
"""
Test Python ↔ Nim ↔ GUI communication for tribal environment.
This tests the core requirement: Python sends actions, Nim processes them and shows GUI.
"""

import sys
import os
import time
from pathlib import Path

def test_basic_communication():
    """Test 1: Python → Nim environment communication (no GUI)"""
    print("=" * 60)
    print("🧪 TEST 1: Python → Nim Communication (No GUI)")
    print("=" * 60)
    
    try:
        # Setup paths
        tribal_dir = Path(__file__).parent / "tribal"
        bindings_path = tribal_dir / "bindings" / "generated"
        
        if not bindings_path.exists():
            print("🔨 Building tribal bindings...")
            import subprocess
            result = subprocess.run(
                ["bash", "build_bindings.sh"],
                cwd=str(tribal_dir),
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                print(f"❌ Failed to build bindings: {result.stderr}")
                return False
        
        # Import bindings
        sys.path.insert(0, str(bindings_path))
        import tribal
        
        print("✅ Imported tribal bindings")
        
        # Create environment
        config = tribal.default_tribal_config()
        env = tribal.new_tribal_env(config)
        print("✅ Created Nim environment")
        
        # Initialize external NN controller (required for Python control)
        if not tribal.init_external_nn_controller():
            print("❌ Failed to initialize external NN controller")
            return False
        print("✅ Initialized external NN controller")
        
        # Reset environment
        env.reset_env()
        print("✅ Reset environment")
        
        # Test Python → Nim action sending
        print("🎯 Testing Python → Nim action communication...")
        
        for step in range(5):
            # Create test actions (all agents move randomly)
            import random
            actions = []
            for agent in range(15):  # 15 agents
                action_type = 1  # MOVE
                direction = random.randint(0, 3)  # Random direction
                actions.extend([action_type, direction])
            
            # Send actions to Nim
            success = tribal.set_external_actions_from_python(actions)
            if not success:
                print(f"❌ Failed to send actions at step {step}")
                return False
            
            # Step environment
            step_success = env.step(actions)
            if not step_success:
                print(f"❌ Environment step failed at step {step}")
                return False
            
            # Get results back from Nim
            rewards = env.get_rewards()
            observations = env.get_token_observations()
            
            print(f"  Step {step}: Sent {len(actions)} actions, got {len(rewards)} rewards, {len(observations)} obs tokens")
        
        print("✅ SUCCESS: Python ↔ Nim communication working!")
        return True
        
    except Exception as e:
        print(f"❌ Communication test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gradual_opengl():
    """Test 2: Gradual OpenGL initialization to find crash point"""
    print("\n" + "=" * 60)
    print("🧪 TEST 2: Gradual OpenGL Initialization")
    print("=" * 60)
    
    try:
        # Change to tribal directory
        tribal_dir = Path(__file__).parent / "tribal"
        old_cwd = os.getcwd()
        os.chdir(tribal_dir)
        
        # Build safe viewer if it doesn't exist
        safe_viewer_so = tribal_dir / "tribal_nimpy_safe_viewer.so"
        if not safe_viewer_so.exists():
            print("🔨 Building safe viewer...")
            import subprocess
            result = subprocess.run([
                "nim", "c", "--app:lib", "--out:tribal_nimpy_safe_viewer.so", 
                "-d:release", "src/tribal_nimpy_safe_viewer.nim"
            ], cwd=str(tribal_dir), capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"❌ Failed to build safe viewer: {result.stderr}")
                return False
        
        # Import safe viewer
        sys.path.insert(0, str(tribal_dir))
        import tribal_nimpy_safe_viewer as safe_viewer
        print("✅ Imported safe viewer")
        
        # Test 1: Window creation only
        print("🔧 Step 1: Testing window creation...")
        if not safe_viewer.initWindowOnly():
            print("❌ Window creation failed")
            return False
        print("✅ Window created successfully")
        
        status = safe_viewer.getInitializationStatus()
        print(f"  Status: {status}")
        
        # Test 2: OpenGL context
        print("🔧 Step 2: Testing OpenGL context...")
        if not safe_viewer.initOpenGLContext():
            print("❌ OpenGL context creation failed - THIS IS THE CRASH POINT")
            return False
        print("✅ OpenGL context created successfully")
        
        # Test 3: Panel initialization
        print("🔧 Step 3: Testing panel initialization...")
        if not safe_viewer.initPanels():
            print("❌ Panel initialization failed")
            return False
        print("✅ Panels initialized successfully")
        
        # Test 4: Minimal rendering
        print("🔧 Step 4: Testing minimal rendering...")
        if not safe_viewer.renderFrameMinimal():
            print("❌ Minimal rendering failed")
            return False
        print("✅ Minimal rendering works")
        
        # Test 5: Asset loading
        print("🔧 Step 5: Testing asset loading...")
        if not safe_viewer.loadAssetsSafe():
            print("❌ Asset loading failed")
            return False
        print("✅ Assets loaded successfully")
        
        # Cleanup
        safe_viewer.closeVisualization()
        print("✅ All OpenGL steps successful!")
        return True
        
    except Exception as e:
        print(f"❌ Gradual OpenGL test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        os.chdir(old_cwd)

def test_full_communication_loop():
    """Test 3: Full Python → Nim → GUI → Python loop"""
    print("\n" + "=" * 60)
    print("🧪 TEST 3: Full Communication Loop with GUI")
    print("=" * 60)
    
    try:
        # Setup both bindings
        tribal_dir = Path(__file__).parent / "tribal"
        old_cwd = os.getcwd()
        os.chdir(tribal_dir)
        
        # Import both modules
        bindings_path = tribal_dir / "bindings" / "generated"
        sys.path.insert(0, str(bindings_path))
        sys.path.insert(0, str(tribal_dir))
        
        import tribal
        import tribal_nimpy_safe_viewer as viewer
        
        print("✅ Both modules imported")
        
        # Create environment
        config = tribal.default_tribal_config()
        env = tribal.new_tribal_env(config)
        
        # Initialize communication
        if not tribal.init_external_nn_controller():
            print("❌ Failed to init NN controller")
            return False
        
        env.reset_env()
        print("✅ Environment ready")
        
        # Initialize visualization
        if not viewer.initVisualizationSafe():
            print("❌ Visualization init failed")
            return False
        print("✅ Visualization ready")
        
        # Load assets
        if not viewer.loadAssetsSafe():
            print("❌ Asset loading failed")
            return False
        print("✅ Assets loaded")
        
        # Run communication loop
        print("🎮 Starting Python → Nim → GUI → Python loop...")
        
        for step in range(10):
            print(f"\n--- Step {step} ---")
            
            # 1. Get observations from Nim
            observations = env.get_token_observations()
            print(f"✅ Got {len(observations)} observation tokens from Nim")
            
            # 2. Generate actions in Python (simulate neural network)
            import random
            actions = []
            for agent in range(15):
                action_type = random.choice([0, 1])  # NOOP or MOVE
                arg = random.randint(0, 3) if action_type == 1 else 0
                actions.extend([action_type, arg])
            
            # 3. Send actions to Nim
            if not tribal.set_external_actions_from_python(actions):
                print("❌ Failed to send actions")
                return False
            print(f"✅ Sent {len(actions)} actions to Nim")
            
            # 4. Step Nim environment
            if not env.step(actions):
                print("❌ Nim environment step failed")
                return False
            print("✅ Nim environment stepped")
            
            # 5. Update GUI
            if not viewer.renderFrameMinimal():
                print("❌ GUI render failed")
                return False
            print("✅ GUI updated")
            
            # 6. Get results back
            rewards = env.get_rewards()
            print(f"✅ Got {len(rewards)} rewards back from Nim")
            
            # Small delay
            time.sleep(0.1)
            
            # Check if window closed
            if not viewer.isWindowOpen():
                print("🪟 Window was closed by user")
                break
        
        # Cleanup
        viewer.closeVisualization()
        print("🎉 SUCCESS: Full communication loop working!")
        return True
        
    except Exception as e:
        print(f"❌ Full communication test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        os.chdir(old_cwd)

def main():
    """Run all communication tests"""
    print("🔬 Python ↔ Nim ↔ GUI Communication Test Suite")
    print("🎯 Goal: Enable Python to send actions to Nim and see results in GUI")
    
    tests = [
        ("Basic Python → Nim Communication", test_basic_communication),
        ("Gradual OpenGL Initialization", test_gradual_opengl),
        ("Full Communication Loop", test_full_communication_loop),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
            
            if result:
                print(f"✅ {test_name}: SUCCESS")
            else:
                print(f"❌ {test_name}: FAILED")
                # Don't continue if basic communication fails
                if test_func == test_basic_communication:
                    print("⚠️  Skipping remaining tests due to basic communication failure")
                    break
                
        except Exception as e:
            print(f"💥 {test_name}: CRASHED - {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 COMMUNICATION TEST RESULTS")
    print("=" * 60)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {test_name}")
    
    # Recommendations
    print(f"\n🎯 NEXT STEPS:")
    
    if len(results) > 0 and results[0][1]:  # Basic communication works
        print("✅ Python ↔ Nim communication is working")
        
        if len(results) > 1:
            if results[1][1]:  # OpenGL works
                print("✅ OpenGL initialization works")
                if len(results) > 2 and results[2][1]:  # Full loop works
                    print("🎉 SOLUTION READY: Full communication working!")
                else:
                    print("🔧 Debug needed: Full loop integration")
            else:
                print("💡 ISSUE IDENTIFIED: OpenGL context creation crashes")
                print("   Recommendation: Check OpenGL drivers, run on different system, or use headless mode")
    else:
        print("❌ Fix basic Python ↔ Nim communication first")
    
    return results

if __name__ == "__main__":
    main()