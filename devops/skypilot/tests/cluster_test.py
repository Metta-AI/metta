#!/usr/bin/env -S uv run
"""
Test script for comprehensive cluster validation.

Tests exit conditions, NCCL communication, and restart functionality
across different cluster configurations.

Usage:
    cluster_test.py launch      # Launch all cluster validation tests
    cluster_test.py check       # Check test results
    cluster_test.py check -l    # Check with detailed logs
"""

import sys

from devops.skypilot.utils.testing_helpers import (
    BaseTestRunner,
    SkyPilotTestLauncher,
    TestCondition,
)
from metta.common.util.text_styles import bold, cyan, yellow

# Test matrix configuration
NODE_CONFIGS = [1, 2, 4]

# Exit condition tests
EXIT_CONDITIONS = {
    "normal_completion": TestCondition(
        name="Normal Completion",
        extra_args=["trainer.total_timesteps=50000"],
        description="Exit normally after training completes",
    ),
    "heartbeat_timeout": TestCondition(
        name="Heartbeat Timeout",
        extra_args=["-hb", "1"],
        description="Exit based on missing heartbeats (1 second timeout)",
    ),
    "runtime_timeout": TestCondition(
        name="Runtime Timeout",
        extra_args=["-t", "0.03"],
        description="Exit based on timeout (0.03 hours = 1.8 minutes)",
    ),
    "cmd_fails": TestCondition(
        name="Invalid Tool Parameters",
        extra_args=["evaluator.epoch_interval=1", "trainer.checkpoint.checkpoint_interval=10"],
        description="Exit when command fails due to invalid parameters",
    ),
}

# Base configuration
BASE_MODULE = "experiments.recipes.arena_basic_easy_shaped.train"
BASE_ARGS = ["--no-spot", "--gpus=4"]


class ClusterTestRunner(BaseTestRunner):
    """Test runner for comprehensive cluster validation tests."""

    def __init__(self):
        super().__init__(
            prog_name="cluster_test.py",
            description="Comprehensive cluster validation: exit conditions, NCCL, and restart tests",
            default_output_file="cluster_test_jobs.json",
            default_base_name="cluster_test",
            test_type="Cluster Test",
        )

    def get_launch_description(self) -> str:
        """Get the launch subcommand description."""
        return "Launch comprehensive cluster validation tests"

    def get_launch_epilog(self) -> str:
        """Get the launch subcommand epilog with examples."""
        return """
This launches comprehensive cluster validation tests:
  - Exit condition tests: normal completion, heartbeat timeout, runtime timeout, cmd failures
  - NCCL communication tests: validate GPU communication across cluster configs
  - Restart tests: validate job restart functionality

Test matrix:
  - 3 node configurations: 1, 2, 4 nodes
  - 4 exit conditions + NCCL test + restart test = 6 test types per config
  - Total jobs: 18 (3 configs × 6 tests)

Examples:
  %(prog)s                      # Launch all validation tests
  %(prog)s --base-name mytest   # Launch with custom base name
  %(prog)s --skip-git-check     # Launch without git validation
        """

    def launch_tests(self, args):
        """Launch comprehensive cluster validation tests."""
        # Create launcher
        launcher = SkyPilotTestLauncher(base_name=args.base_name, skip_git_check=args.skip_git_check)

        # Check git state
        if not launcher.check_git_state():
            sys.exit(1)

        # Show test matrix
        print(f"\n{bold('=== Comprehensive Cluster Validation Tests ===')}")
        print(f"{cyan('Node configurations:')} {NODE_CONFIGS}")
        print(f"\n{cyan('Exit condition tests:')}")
        for _key, condition in EXIT_CONDITIONS.items():
            print(f"  • {yellow(condition.name)}: {condition.description}")
        print(f"\n{cyan('Additional tests per config:')}")
        print(f"  • {yellow('NCCL Communication')}: Validate GPU communication")
        print(f"  • {yellow('Restart Functionality')}: Validate job restart")

        total_jobs = len(NODE_CONFIGS) * (len(EXIT_CONDITIONS) + 2)
        print(f"\n{cyan('Total jobs to launch:')} {total_jobs}")
        print(f"{cyan('Output file:')} {args.output_file}")

        # Launch exit condition tests
        for nodes in NODE_CONFIGS:
            for condition_key, condition in EXIT_CONDITIONS.items():
                run_name = launcher.generate_run_name(f"{nodes}n_{condition_key}")

                test_config = {
                    "test_type": "exit_condition",
                    "nodes": nodes,
                    "condition": condition_key,
                    "condition_name": condition.name,
                    "condition_description": condition.description,
                }

                launcher.launch_job(
                    module=BASE_MODULE,
                    run_name=run_name,
                    base_args=BASE_ARGS + ["--nodes", str(nodes)],
                    extra_args=condition.extra_args,
                    test_config=test_config,
                )

            # Launch NCCL test for this cluster config
            run_name = launcher.generate_run_name(f"{nodes}n_nccl")
            test_config = {
                "test_type": "nccl",
                "nodes": nodes,
                "description": "NCCL communication test",
            }
            launcher.launch_job(
                module="devops.skypilot.tools.nccl",
                run_name=run_name,
                base_args=["--no-spot", "--nodes", str(nodes), "--gpus", "4"],
                extra_args=[],
                test_config=test_config,
            )

            # Launch restart test for this cluster config
            run_name = launcher.generate_run_name(f"{nodes}n_restart")
            test_config = {
                "test_type": "restart",
                "nodes": nodes,
                "description": "Job restart functionality test",
            }
            launcher.launch_job(
                module="devops.skypilot.tools.restart_test",
                run_name=run_name,
                base_args=["--no-spot", "--nodes", str(nodes), "-t", "0.1"],  # 6 minute timeout
                extra_args=[],
                test_config=test_config,
            )

        # Save results
        output_path = launcher.save_results(args.output_file)
        print(f"{cyan('Results saved to:')} {output_path.absolute()}")

        # Print summary
        launcher.print_summary()

        # Exit with error if any launches failed
        if launcher.failed_launches:
            sys.exit(1)


def main():
    """Main entry point."""
    runner = ClusterTestRunner()
    runner.run()


if __name__ == "__main__":
    main()
