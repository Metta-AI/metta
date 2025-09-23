#!/usr/bin/env -S uv run
"""
Test script for validating cluster configurations and exit conditions.

Usage:
    cluster_test.py launch      # Launch 3x3 matrix test jobs
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

TEST_CONDITIONS = {
    "normal_completion": TestCondition(
        name="Normal Completion",
        extra_args=["trainer.total_timesteps=50000"],
        description="Exit normally after training completes",
        ci=False,
    ),
    "heartbeat_timeout": TestCondition(
        name="Heartbeat Timeout",
        extra_args=["-hb", "1"],
        description="Exit based on missing heartbeats (1 second timeout)",
        ci=False,
    ),
    "runtime_timeout": TestCondition(
        name="Runtime Timeout",
        extra_args=["-t", "0.03"],
        description="Exit based on timeout (0.03 hours = 1.8 minutes)",
        ci=True,
    ),
    "cmd_fails": TestCondition(
        name="Invalid Tool Parameters",  # deliberately invalid: evaluate interval must be >= checkpoint interval
        extra_args=["evaluator.epoch_interval=1", "trainer.checkpoint.checkpoint_interval=10"],
        description="Exit when command fails due to invalid parameters",
        ci=True,
    ),
}

# Base configuration
BASE_MODULE = "experiments.recipes.arena_basic_easy_shaped.train"
BASE_ARGS = ["--no-spot", "--gpus=4"]


class ClusterTestRunner(BaseTestRunner):
    """Test runner for cluster configuration tests."""

    def __init__(self):
        super().__init__(
            prog_name="cluster_test.py",
            description="Cluster configuration and exit condition test",
            default_output_file="cluster_test_jobs.json",
            default_base_name="cluster_test",
            test_type="Cluster Test",
        )

    def get_launch_description(self) -> str:
        """Get the launch subcommand description."""
        return "Launch a 3x3 matrix of cluster test jobs"

    def get_launch_epilog(self) -> str:
        """Get the launch subcommand epilog with examples."""
        return """
This launches a 3x3 test matrix:
  - 3 node configurations: 1, 2, 4 nodes
  - 3 test conditions: normal completion, heartbeat timeout, runtime timeout
  - CI tests are enabled for runtime timeout jobs

Examples:
  %(prog)s                      # Launch 9 test jobs
  %(prog)s --base-name mytest   # Launch with custom base name
  %(prog)s --skip-git-check     # Launch without git validation
        """

    def launch_tests(self, args):
        """Launch cluster test jobs."""
        # Create launcher
        launcher = SkyPilotTestLauncher(base_name=args.base_name, skip_git_check=args.skip_git_check)

        # Check git state
        if not launcher.check_git_state():
            sys.exit(1)

        # Show test matrix
        print(f"\n{bold('=== Skypilot Cluster Test Matrix ===')}")
        print(f"{cyan('Node configurations:')} {NODE_CONFIGS}")
        print(f"{cyan('Test conditions:')}")
        for _key, condition in TEST_CONDITIONS.items():
            print(f"  • {yellow(condition.name)}: {condition.description}")
        print(f"\n{cyan('Total jobs to launch:')} {len(NODE_CONFIGS) * len(TEST_CONDITIONS)}")
        print(f"{cyan('Output file:')} {args.output_file}")

        # Launch jobs
        for nodes in NODE_CONFIGS:
            for condition_key, condition in TEST_CONDITIONS.items():
                enable_ci_tests = condition.ci

                # Generate unique run name
                suffix = f"{nodes}n_{condition_key}"
                if enable_ci_tests:
                    suffix += "_ci"
                run_name = launcher.generate_run_name(suffix)

                # Test config for tracking
                test_config = {
                    "nodes": nodes,
                    "condition": condition_key,
                    "condition_name": condition.name,
                    "condition_description": condition.description,
                    "ci_tests_enabled": enable_ci_tests,
                }

                # Launch the job
                launcher.launch_job(
                    module=BASE_MODULE,
                    run_name=run_name,
                    base_args=BASE_ARGS + ["--nodes", str(nodes)],
                    extra_args=condition.extra_args,
                    test_config=test_config,
                    enable_ci_tests=enable_ci_tests,
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
