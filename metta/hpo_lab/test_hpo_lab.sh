#!/bin/bash
# HPO Lab Test Script
# This script demonstrates the various ways to use the HPO Lab

echo "=== HPO Lab Test Suite ==="
echo ""

# Test 1: Direct training function
echo "1. Testing direct training function..."
uv run python -c "
from metta.hpo_lab.recipes.lunarlander import train
metrics = train(run='test_direct', total_timesteps=10000, verbose=1)
print(f'Direct training completed: {metrics[\"final_mean_reward\"]:.2f}')
"

echo ""
echo "2. Testing TrainSB3GymEnvTool via CLI..."
# Test 2: Tool invocation via CLI
uv run ./tools/run.py metta.hpo_lab.recipes.lunarlander.train_sb3_tool \
    run=test_tool \
    total_timesteps=10000 \
    verbose=1

echo ""
echo "3. Testing with different environment (CartPole)..."
# Test 3: Different environment
uv run ./tools/run.py metta.hpo_lab.recipes.lunarlander.train_sb3_tool \
    run=test_cartpole \
    env_id="CartPole-v1" \
    total_timesteps=10000 \
    learning_rate=0.001

echo ""
echo "4. Testing evaluation function..."
# Test 4: Evaluation
uv run python -c "
from metta.hpo_lab.recipes.lunarlander import evaluate
metrics = evaluate(n_eval_episodes=10)
print(f'Random policy evaluation: {metrics[\"mean_reward\"]:.2f} ± {metrics[\"std_reward\"]:.2f}')
"

echo ""
echo "=== All tests completed ==="
echo ""
echo "Available tools in HPO Lab:"
echo "  - train(): Direct training function"
echo "  - train_sb3_tool(): Tool-based training via CLI"
echo "  - evaluate(): Evaluation function"
echo "  - ray_sweep(): Full hyperparameter sweep (100 trials)"
echo "  - mini_sweep(): Quick testing sweep (10 trials)"
echo ""
echo "Example sweep usage:"
echo "  uv run ./tools/run.py metta.hpo_lab.recipes.lunarlander.mini_sweep sweep_name=test_sweep"