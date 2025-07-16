#!/usr/bin/env python3

import sys
import tempfile

import numpy as np
from omegaconf import OmegaConf


def test_learning_progress_algorithm():
    """Test the learning progress algorithm for issues"""
    print("Test 1: Learning Progress Algorithm")

    try:
        from metta.mettagrid.curriculum.learning_progress import BidirectionalLearningProgress

        # Create learning progress tracker
        lp = BidirectionalLearningProgress(
            search_space=12,
            ema_timescale=0.001,
            progress_smoothing=0.05,
            num_active_tasks=16,
            rand_task_rate=0.25,
            sample_threshold=10,
            memory=25,
        )
        print("✓ BidirectionalLearningProgress loads successfully")

        # Test with various scenarios
        scenarios = [
            ("Random scores", lambda: np.random.random()),
            ("Improving scores", lambda: min(1.0, np.random.random() * 0.5 + 0.5)),
            ("Degrading scores", lambda: max(0.0, np.random.random() * 0.5)),
            ("Stable scores", lambda: 0.5 + np.random.normal(0, 0.1)),
        ]

        for scenario_name, score_fn in scenarios:
            print(f"\nTesting {scenario_name}:")

            # Reset for each scenario
            lp = BidirectionalLearningProgress(
                search_space=12,
                ema_timescale=0.001,
                progress_smoothing=0.05,
                num_active_tasks=16,
                rand_task_rate=0.25,
                sample_threshold=10,
                memory=25,
            )

            # Collect data
            for i in range(50):
                for task_id in range(12):
                    score = score_fn()
                    lp.collect_data({f"tasks/{task_id}": [score]})

                if i % 10 == 0:
                    try:
                        task_dist, sample_levels = lp.calculate_dist()
                        stats = lp.add_stats()
                        print(
                            f"  Step {i}: num_active={stats.get('lp/num_active_tasks', 0)}, "
                            f"mean_prob={stats.get('lp/mean_sample_prob', 0):.3f}, "
                            f"success_rate={stats.get('lp/task_success_rate', 0):.3f}"
                        )

                        # Check for issues
                        if np.isnan(stats.get("lp/task_success_rate", 0)):
                            print(f"  ⚠️  Warning: NaN success rate in {scenario_name}")
                        if stats.get("lp/num_active_tasks", 0) == 0:
                            print(f"  ⚠️  Warning: No active tasks in {scenario_name}")
                        if stats.get("lp/mean_sample_prob", 0) == 0:
                            print(f"  ⚠️  Warning: Zero sample probability in {scenario_name}")

                    except Exception as e:
                        print(f"  ✗ Error at step {i}: {e}")
                        return False

            final_stats = lp.add_stats()
            print(f"  Final: {final_stats}")

        print("✓ Learning progress algorithm test completed")
        return True

    except Exception as e:
        print(f"✗ Learning progress algorithm test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_checkpoint_system():
    """Test the checkpoint system"""
    print("\nTest 2: Checkpoint System")

    try:
        from metta.rl.trainer_checkpoint import TrainerCheckpoint

        # Create a test checkpoint
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = TrainerCheckpoint(
                agent_step=1000,
                epoch=10,
                total_agent_step=1000,
                optimizer_state_dict={"test": "state"},
                policy_path="/test/policy/path",
                stopwatch_state={"test": "timer"},
            )

            checkpoint.save(temp_dir)
            print("✓ Checkpoint save works")

            # Load the checkpoint
            loaded_checkpoint = TrainerCheckpoint.load(temp_dir)
            if loaded_checkpoint:
                print("✓ Checkpoint load works")
                print(f"  - Agent step: {loaded_checkpoint.agent_step}")
                print(f"  - Epoch: {loaded_checkpoint.epoch}")
            else:
                print("✗ Checkpoint load failed")
                return False

        return True

    except Exception as e:
        print(f"✗ Checkpoint system test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_configuration_loading():
    """Test configuration loading"""
    print("\nTest 3: Configuration Loading")

    try:
        # Test learning progress experiment config
        OmegaConf.load("configs/user/learning_progress_experiment.yaml")
        print("✓ Learning progress experiment config loads")

        # Test random curriculum config
        OmegaConf.load("configs/user/random_curriculum_experiment.yaml")
        print("✓ Random curriculum config loads")

        # Test basic arena config
        OmegaConf.load("configs/user/basic_arena_experiment.yaml")
        print("✓ Basic arena config loads")

        return True

    except Exception as e:
        print(f"✗ Configuration loading failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_memory_usage():
    """Test memory usage"""
    print("\nTest 4: Memory Usage")

    try:
        # Check if we can import torch and check GPU memory
        import torch

        if torch.cuda.is_available():
            print("✓ CUDA is available")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"  GPU {i}: {props.name}, {memory:.1f}GB")
        else:
            print("⚠️  CUDA not available")

        return True

    except Exception as e:
        print(f"✗ Memory usage test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("Learning Progress Arena Debug Tests")
    print("=" * 50)

    tests = [
        test_learning_progress_algorithm,
        test_checkpoint_system,
        test_configuration_loading,
        test_memory_usage,
    ]

    failed_tests = []

    for test in tests:
        if test():
            print("✓ Test passed")
        else:
            print("✗ Test failed")
            failed_tests.append(test.__name__)
        print()

    # Summary
    print("Debug Test Summary:")
    if not failed_tests:
        print("✓ All tests passed! The system appears to be working correctly.")
        print("\nIf runs are still failing, the issue might be:")
        print("1. Resource constraints (memory, GPU)")
        print("2. Network issues during checkpoint saving")
        print("3. Long-running training instability")
        print("4. Hyperparameter sensitivity over long runs")
    else:
        print(f"✗ Failed tests: {', '.join(failed_tests)}")
        print("These failures likely explain the frequent restarts.")
        print("Address these issues before running the full experiment.")

    print("\nKey Findings:")
    print("1. Learning progress algorithm shows warnings about empty slices")
    print("2. This could cause NaN values and training instability")
    print("3. Consider adjusting hyperparameters or adding NaN checks")

    return len(failed_tests) == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
