#!/bin/bash

# Learning Progress Arena Experiment Debug Script
# This script helps diagnose why runs are failing and recovering frequently

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Test 1: Environment Loading
test_environment_loading() {
    log_info "Test 1: Environment Loading"

    python -c "
import sys
from omegaconf import OmegaConf

try:
    from metta.mettagrid.mettagrid_env import MettaGridEnv

    # Test basic_easy environment
    cfg = OmegaConf.load('configs/env/mettagrid/arena/basic_easy.yaml')
    env = MettaGridEnv(cfg)
    obs = env.reset()
    print('✓ Basic easy environment loads successfully')

    # Test all arena environments
    arena_configs = [
        'configs/env/mettagrid/arena/basic.yaml',
        'configs/env/mettagrid/arena/basic_easy.yaml',
        'configs/env/mettagrid/arena/basic_easy_shaped.yaml',
        'configs/env/mettagrid/arena/combat.yaml',
        'configs/env/mettagrid/arena/combat_easy.yaml',
        'configs/env/mettagrid/arena/combat_easy_shaped.yaml',
        'configs/env/mettagrid/arena/advanced.yaml',
        'configs/env/mettagrid/arena/advanced_easy.yaml',
        'configs/env/mettagrid/arena/advanced_easy_shaped.yaml',
        'configs/env/mettagrid/arena/tag.yaml',
        'configs/env/mettagrid/arena/tag_easy.yaml',
        'configs/env/mettagrid/arena/tag_easy_shaped.yaml'
    ]

    for config_path in arena_configs:
        try:
            cfg = OmegaConf.load(config_path)
            env = MettaGridEnv(cfg)
            obs = env.reset()
            print(f'✓ {config_path} loads successfully')
        except Exception as e:
            print(f'✗ {config_path} failed: {e}')

except Exception as e:
    print(f'✗ Environment loading failed: {e}')
    sys.exit(1)
"
}

# Test 2: Curriculum Loading
test_curriculum_loading() {
    log_info "Test 2: Curriculum Loading"

    python -c "
import sys
import numpy as np
from omegaconf import OmegaConf

try:
    from metta.mettagrid.curriculum.learning_progress import LearningProgressCurriculum
    from metta.mettagrid.curriculum.random import RandomCurriculum

    # Test learning progress curriculum
    tasks = {f'task_{i}': 1.0 for i in range(12)}
    lp_curriculum = LearningProgressCurriculum(
        tasks=tasks,
        ema_timescale=0.001,
        progress_smoothing=0.05,
        num_active_tasks=16,
        rand_task_rate=0.25,
        sample_threshold=10,
        memory=25
    )
    print('✓ Learning progress curriculum loads successfully')

    # Test random curriculum
    random_curriculum = RandomCurriculum(tasks)
    print('✓ Random curriculum loads successfully')

    # Test curriculum sampling
    for i in range(10):
        task = lp_curriculum.get_task()
        score = np.random.random()
        lp_curriculum.complete_task(task.id, score)

    stats = lp_curriculum.stats()
    print(f'✓ Curriculum sampling works, stats: {stats}')

except Exception as e:
    print(f'✗ Curriculum loading failed: {e}')
    sys.exit(1)
"
}

# Test 3: Configuration Loading
test_configuration_loading() {
    log_info "Test 3: Configuration Loading"

    python -c "
import sys
from omegaconf import OmegaConf

try:
    # Test learning progress experiment config
    cfg = OmegaConf.load('configs/user/learning_progress_experiment.yaml')
    print('✓ Learning progress experiment config loads')
    print(f'  - Total timesteps: {cfg.trainer.total_timesteps}')
    print(f'  - Num workers: {cfg.trainer.num_workers}')
    print(f'  - Batch size: {cfg.trainer.batch_size}')
    print(f'  - EMA timescale: {cfg.ema_timescale}')

    # Test random curriculum config
    cfg = OmegaConf.load('configs/user/random_curriculum_experiment.yaml')
    print('✓ Random curriculum config loads')

    # Test basic arena config
    cfg = OmegaConf.load('configs/user/basic_arena_experiment.yaml')
    print('✓ Basic arena config loads')

except Exception as e:
    print(f'✗ Configuration loading failed: {e}')
    sys.exit(1)
"
}

# Test 4: Memory Usage Check
test_memory_usage() {
    log_info "Test 4: Memory Usage Check"

    # Check GPU memory
    if command -v nvidia-smi &> /dev/null; then
        echo "GPU Memory Status:"
        nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | while IFS=',' read -r used total; do
            usage_pct=$((used * 100 / total))
            echo "  GPU: ${used}MB / ${total}MB (${usage_pct}%)"
        done
    else
        log_warning "nvidia-smi not available"
    fi

    # Check system memory
    echo "System Memory:"
    free -h | grep -E "Mem|Swap"
}

# Test 5: Learning Progress Algorithm
test_learning_progress_algorithm() {
    log_info "Test 5: Learning Progress Algorithm"

    python -c "
import sys
import numpy as np
import logging
logging.basicConfig(level=logging.WARNING)

try:
    from metta.mettagrid.curriculum.learning_progress import LearningProgressCurriculum

    # Create curriculum with test tasks
    tasks = {f'task_{i}': 1.0 for i in range(12)}
    curriculum = LearningProgressCurriculum(
        tasks=tasks,
        ema_timescale=0.001,
        progress_smoothing=0.05,
        num_active_tasks=16,
        rand_task_rate=0.25,
        sample_threshold=10,
        memory=25
    )

    print('Testing learning progress algorithm...')

    # Simulate training with various scenarios
    scenarios = [
        ('Random scores', lambda: np.random.random()),
        ('Improving scores', lambda: min(1.0, np.random.random() * 0.5 + 0.5)),
        ('Degrading scores', lambda: max(0.0, np.random.random() * 0.5)),
        ('Stable scores', lambda: 0.5 + np.random.normal(0, 0.1)),
    ]

    for scenario_name, score_fn in scenarios:
        print(f'\\nTesting {scenario_name}:')

        # Reset curriculum for each scenario
        curriculum = LearningProgressCurriculum(
            tasks=tasks,
            ema_timescale=0.001,
            progress_smoothing=0.05,
            num_active_tasks=16,
            rand_task_rate=0.25,
            sample_threshold=10,
            memory=25
        )

        for i in range(50):
            task = curriculum.get_task()
            score = score_fn()
            curriculum.complete_task(task.id, score)

            if i % 10 == 0:
                stats = curriculum.stats()
                print(f'  Step {i}: num_active={stats.get(\"lp/num_active_tasks\", 0)}, '
                      f'mean_prob={stats.get(\"lp/mean_sample_prob\", 0):.3f}, '
                      f'success_rate={stats.get(\"lp/task_success_rate\", 0):.3f}')

        # Check for potential issues
        final_stats = curriculum.stats()
        if final_stats.get('lp/num_active_tasks', 0) == 0:
            print(f'  ⚠️  Warning: No active tasks in {scenario_name}')
        if final_stats.get('lp/mean_sample_prob', 0) == 0:
            print(f'  ⚠️  Warning: Zero sample probability in {scenario_name}')
        if np.isnan(final_stats.get('lp/task_success_rate', 0)):
            print(f'  ⚠️  Warning: NaN success rate in {scenario_name}')

    print('\\n✓ Learning progress algorithm test completed')

except Exception as e:
    print(f'✗ Learning progress algorithm test failed: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"
}

# Test 6: Checkpoint System
test_checkpoint_system() {
    log_info "Test 6: Checkpoint System"

    # Create test checkpoint
    python -c "
import torch
import tempfile
import os
from metta.rl.trainer_checkpoint import TrainerCheckpoint

try:
    # Create a test checkpoint
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint = TrainerCheckpoint(
            agent_step=1000,
            epoch=10,
            total_agent_step=1000,
            optimizer_state_dict={'test': 'state'},
            policy_path='/test/policy/path',
            stopwatch_state={'test': 'timer'}
        )

        checkpoint.save(temp_dir)
        print('✓ Checkpoint save works')

        # Load the checkpoint
        loaded_checkpoint = TrainerCheckpoint.load(temp_dir)
        if loaded_checkpoint:
            print('✓ Checkpoint load works')
            print(f'  - Agent step: {loaded_checkpoint.agent_step}')
            print(f'  - Epoch: {loaded_checkpoint.epoch}')
        else:
            print('✗ Checkpoint load failed')

except Exception as e:
    print(f'✗ Checkpoint system test failed: {e}')
    import traceback
    traceback.print_exc()
"
}

# Test 7: Small Training Run
test_small_training_run() {
    log_info "Test 7: Small Training Run"

    # Run a very small training test
    echo "Running small training test (1M timesteps)..."

    ./devops/skypilot/launch.py train \
        --gpus=1 \
        --nodes=1 \
        --no-spot \
        run="debug.test.small" \
        --config configs/user/learning_progress_experiment.yaml \
        trainer.total_timesteps=1_000_000 \
        trainer.num_workers=2 \
        trainer.batch_size=16384 \
        trainer.checkpoint.checkpoint_interval=25 \
        trainer.simulation.evaluate_interval=50

    if [ $? -eq 0 ]; then
        log_success "Small training run completed successfully"
    else
        log_error "Small training run failed"
        return 1
    fi
}

# Test 8: Hyperparameter Sensitivity
test_hyperparameter_sensitivity() {
    log_info "Test 8: Hyperparameter Sensitivity"

    python -c "
import sys
import numpy as np
from metta.mettagrid.curriculum.learning_progress import LearningProgressCurriculum

try:
    tasks = {f'task_{i}': 1.0 for i in range(12)}

    # Test different hyperparameter combinations
    test_configs = [
        {'ema_timescale': 0.0001, 'progress_smoothing': 0.01, 'num_active_tasks': 8},
        {'ema_timescale': 0.001, 'progress_smoothing': 0.05, 'num_active_tasks': 16},
        {'ema_timescale': 0.01, 'progress_smoothing': 0.1, 'num_active_tasks': 24},
        {'ema_timescale': 0.001, 'progress_smoothing': 0.2, 'num_active_tasks': 16},
    ]

    for i, config in enumerate(test_configs):
        print(f'\\nTesting config {i+1}: {config}')

        curriculum = LearningProgressCurriculum(
            tasks=tasks,
            ema_timescale=config['ema_timescale'],
            progress_smoothing=config['progress_smoothing'],
            num_active_tasks=config['num_active_tasks'],
            rand_task_rate=0.25,
            sample_threshold=10,
            memory=25
        )

        # Run some steps
        for step in range(30):
            task = curriculum.get_task()
            score = np.random.random()
            curriculum.complete_task(task.id, score)

            if step % 10 == 0:
                stats = curriculum.stats()
                print(f'  Step {step}: active_tasks={stats.get(\"lp/num_active_tasks\", 0)}, '
                      f'mean_prob={stats.get(\"lp/mean_sample_prob\", 0):.3f}')

        final_stats = curriculum.stats()
        print(f'  Final: active_tasks={final_stats.get(\"lp/num_active_tasks\", 0)}, '
              f'mean_prob={final_stats.get(\"lp/mean_sample_prob\", 0):.3f}')

    print('\\n✓ Hyperparameter sensitivity test completed')

except Exception as e:
    print(f'✗ Hyperparameter sensitivity test failed: {e}')
    import traceback
    traceback.print_exc()
"
}

# Main test runner
main() {
    log_info "Starting Learning Progress Arena Debug Tests"
    log_info "This will help identify why runs are failing and recovering frequently"

    # Run all tests
    tests=(
        test_environment_loading
        test_curriculum_loading
        test_configuration_loading
        test_memory_usage
        test_learning_progress_algorithm
        test_checkpoint_system
        test_hyperparameter_sensitivity
    )

    failed_tests=()

    for test in "${tests[@]}"; do
        log_info "Running $test..."
        if $test; then
            log_success "$test passed"
        else
            log_error "$test failed"
            failed_tests+=("$test")
        fi
        echo
    done

    # Optional: Run small training test (takes longer)
    read -p "Run small training test? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if test_small_training_run; then
            log_success "Small training test passed"
        else
            log_error "Small training test failed"
            failed_tests+=("test_small_training_run")
        fi
    fi

    # Summary
    echo
    log_info "Debug Test Summary:"
    if [ ${#failed_tests[@]} -eq 0 ]; then
        log_success "All tests passed! The system appears to be working correctly."
        log_info "If runs are still failing, the issue might be:"
        log_info "1. Resource constraints (memory, GPU)"
        log_info "2. Network issues during checkpoint saving"
        log_info "3. Long-running training instability"
        log_info "4. Hyperparameter sensitivity over long runs"
    else
        log_error "Failed tests: ${failed_tests[*]}"
        log_info "These failures likely explain the frequent restarts."
        log_info "Address these issues before running the full experiment."
    fi

    echo
    log_info "Next steps:"
    log_info "1. Fix any failed tests above"
    log_info "2. Run with smaller batch sizes if memory issues detected"
    log_info "3. Adjust learning progress hyperparameters if algorithm issues found"
    log_info "4. Increase checkpoint frequency if checkpoint issues detected"
    log_info "5. Monitor the runs more closely with the metrics from the testing guide"
}

# Run the main function
main "$@"
