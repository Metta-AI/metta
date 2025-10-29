#!/usr/bin/env -S uv run
"""Test script for validating NCCL communication across cluster configurations.

Usage:
    nccl_test.py launch      # Launch NCCL test jobs across different configurations
    nccl_test.py check       # Check test results
    nccl_test.py check -l    # Check with detailed logs
"""

import sys

from devops.skypilot.utils.testing_helpers import (
    BaseTestRunner,
    SkyPilotTestLauncher,
)
from metta.common.util.text_styles import bold, cyan, yellow

# NCCL test matrix - different cluster configurations
NCCL_TEST_CONFIGS = [
    {"nodes": 1, "gpus": 4, "description": "Single node, 4 GPUs"},
    {"nodes": 2, "gpus": 4, "description": "2 nodes, 4 GPUs each"},
    {"nodes": 4, "gpus": 4, "description": "4 nodes, 4 GPUs each"},
]

# Base configuration
BASE_ARGS = ["--no-spot"]


class NCCLTestRunner(BaseTestRunner):
    """Test runner for NCCL communication tests."""

    def __init__(self):
        super().__init__(
            prog_name="nccl_test.py",
            description="NCCL communication test across cluster configurations",
            default_output_file="nccl_test_jobs.json",
            default_base_name="nccl_test",
            test_type="NCCL Test",
        )

    def get_launch_description(self) -> str:
        """Get the launch subcommand description."""
        return "Launch NCCL test jobs across different cluster configurations"

    def get_launch_epilog(self) -> str:
        """Get the launch subcommand epilog with examples."""
        return """
This launches NCCL tests across different cluster configurations:
  - 1 node, 4 GPUs
  - 2 nodes, 4 GPUs each
  - 4 nodes, 4 GPUs each

NCCL tests validate GPU communication infrastructure and should be run:
  - Before large/expensive training runs
  - After changing cloud provider, region, or instance type
  - When investigating training failures or hangs

Examples:
  %(prog)s                      # Launch NCCL tests
  %(prog)s --base-name mytest   # Launch with custom base name
  %(prog)s --skip-git-check     # Launch without git validation
        """

    def launch_tests(self, args):
        """Launch NCCL test jobs."""
        # Create launcher
        launcher = SkyPilotTestLauncher(base_name=args.base_name, skip_git_check=args.skip_git_check)

        # Check git state
        if not launcher.check_git_state():
            sys.exit(1)

        # Show test matrix
        print(f"\n{bold('=== NCCL Test Matrix ===')}")
        print(f"{cyan('Test configurations:')}")
        for config in NCCL_TEST_CONFIGS:
            print(f"  • {yellow(config['description'])}: {config['nodes']} nodes x {config['gpus']} GPUs")
        print(f"\n{cyan('Total jobs to launch:')} {len(NCCL_TEST_CONFIGS)}")
        print(f"{cyan('Output file:')} {args.output_file}")

        # Launch jobs
        for config in NCCL_TEST_CONFIGS:
            nodes = config["nodes"]
            gpus = config["gpus"]
            description = config["description"]

            # Generate unique run name
            suffix = f"{nodes}n_{gpus}g"
            run_name = launcher.generate_run_name(suffix)

            # Test config for tracking
            test_config = {
                "nodes": nodes,
                "gpus": gpus,
                "description": description,
                "test_type": "nccl",
            }

            # Launch the job
            launcher.launch_job(
                module="devops.skypilot.tools.nccl",
                run_name=run_name,
                base_args=BASE_ARGS + ["--nodes", str(nodes), "--gpus", str(gpus)],
                extra_args=[],
                test_config=test_config,
                enable_ci_tests=False,
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
    runner = NCCLTestRunner()
    runner.run()


if __name__ == "__main__":
    main()
