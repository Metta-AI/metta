#!/bin/bash
# Test script for the fixed tribal recipe with in-process nimpy

set -e

echo "🧪 Testing Tribal Recipe with In-Process Nimpy"
echo "=============================================="

# Clean up any background processes
pkill -f "nim.*tribal" || true
pkill -f "tools/run.py" || true
pkill -f "uv run" || true

TEST_ID="test_$(date +%H%M%S)"
echo "🏷️  Test ID: $TEST_ID"
echo ""

echo "📋 Available Tests:"
echo "1. Neural Network with NOOP actions (Direct Genny Bindings)"
echo "2. Neural Network with MOVE actions (Direct Genny Bindings)"
echo "3. Built-in AI (no neural network)"
echo "4. Train a policy"
echo "5. Play with trained policy"
echo "6. Performance test: Direct Genny vs Standard Path"
echo ""

# Default to quick test if no argument provided
TEST_TYPE=${1:-"1"}

case $TEST_TYPE in
    "1")
        echo "🧪 Running Test 1: Neural Network with NOOP actions (Direct Genny)"
        echo "   Expected: Agents stay still, '🎯 Using direct genny bindings' messages"
        echo "   Performance: Python lists → SeqInt → Nim (no numpy conversion)"
        echo ""
        uv run ./tools/run.py experiments.recipes.tribal_basic.play --args policy_uri=test_noop
        ;;
    "2")
        echo "🧪 Running Test 2: Neural Network with MOVE actions (Direct Genny)"
        echo "   Expected: Agents move in different directions, direct genny calls"
        echo "   Performance: Optimized Python-to-Nim communication"
        echo ""
        uv run ./tools/run.py experiments.recipes.tribal_basic.play --args policy_uri=test_move
        ;;
    "3")
        echo "🧪 Running Test 3: Built-in AI (Native Nim)"
        echo "   Expected: Nim AI controls agents automatically"
        echo "   Note: Uses built-in controller, no Python neural network"
        echo ""
        uv run ./tools/run.py experiments.recipes.tribal_basic.play
        ;;
    "4")
        echo "🧪 Running Test 4: Train a policy"
        echo "   Expected: Training completes and saves checkpoint"
        echo "   Uses: DirectGennyTribalEnv for optimal performance"
        echo ""
        uv run ./tools/run.py experiments.recipes.tribal_basic.train --overrides run=$TEST_ID trainer.total_timesteps=5000
        echo ""
        echo "✅ Training complete! Checkpoint saved to: ./train_dir/$TEST_ID/$TEST_ID/checkpoints"
        ;;
    "5")
        echo "🧪 Running Test 5: Play with trained policy"
        echo "   First training a policy, then playing with it..."
        echo "   Uses: DirectGennyTribalEnv for both training and play"
        echo ""
        echo "   Step 1: Training with optimized genny bindings..."
        uv run ./tools/run.py experiments.recipes.tribal_basic.train --overrides run=$TEST_ID trainer.total_timesteps=2000
        echo ""
        echo "   Step 2: Playing with trained policy using direct genny..."
        uv run ./tools/run.py experiments.recipes.tribal_basic.play --args policy_uri=file://./train_dir/$TEST_ID/$TEST_ID/checkpoints
        ;;
    "6")
        echo "🧪 Running Test 6: Performance Comparison"
        echo "   Comparing DirectGennyTribalEnv optimization vs standard path"
        echo "   Look for: '🎯 Using direct genny bindings' (optimized) vs '🔄 Using numpy fallback' (standard)"
        echo ""
        
        echo "   🔹 Step 1: Testing optimized direct genny path..."
        uv run ./tools/run.py experiments.recipes.tribal_basic.play --args policy_uri=test_noop
        echo ""
        
        echo "   🔹 Step 2: Built-in AI for comparison..."
        uv run ./tools/run.py experiments.recipes.tribal_basic.play
        echo ""
        
        echo "✅ Performance test complete!"
        echo "   Direct genny bindings eliminate numpy → SeqInt conversion overhead"
        echo "   Python lists are converted directly to SeqInt for optimal performance"
        ;;
    "all")
        echo "🧪 Running all tests..."
        echo ""
        
        echo "Test 1/3: NOOP actions"
        uv run ./tools/run.py experiments.recipes.tribal_basic.play --args policy_uri=test_noop
        echo ""
        
        echo "Test 2/3: MOVE actions"
        uv run ./tools/run.py experiments.recipes.tribal_basic.play --args policy_uri=test_move
        echo ""
        
        echo "Test 3/3: Built-in AI"
        uv run ./tools/run.py experiments.recipes.tribal_basic.play
        echo ""
        
        echo "✅ All quick tests complete!"
        ;;
    *)
        echo "❌ Invalid test type: $TEST_TYPE"
        echo ""
        echo "Usage: $0 [test_number]"
        echo "  1 - Neural Network with NOOP actions (default)"
        echo "  2 - Neural Network with MOVE actions"
        echo "  3 - Built-in AI"
        echo "  4 - Train a policy"
        echo "  5 - Train and play with policy"
        echo "  all - Run tests 1-3"
        exit 1
        ;;
esac

echo ""
echo "🎉 Test complete!"
echo ""
echo "✅ Success indicators to look for:"
echo "   - '🎯 Using TribalGridEnv with direct nimpy interface'"
echo "   - '✅ Environment created with 15 agents'"
echo "   - '🎮 Starting interactive game loop...'"
echo "   - Episode completion with reward totals"
echo "   - No subprocess launching or file communication"