#!/usr/bin/env -S uv run
"""
Test script for running multiple repeats of a single job configuration.

Usage:
    repeat_test.py launch      # Launch 10 repeat jobs
    repeat_test.py check       # Check test results
    repeat_test.py check -l    # Check with detailed logs
"""

import sys

from devops.skypilot.utils.testing_helpers import (
    BaseTestRunner,
    SkyPilotTestLauncher,
)
from metta.common.util.text_styles import bold, cyan, yellow

# Configuration
NUM_REPEATS = 10
BASE_MODULE = "experiments.recipes.icl_resource_chain.train"
BASE_ARGS = ["--no-spot", "--gpus=4"]
RUN_PREFIX = "icl.smallrooms.test"


class RepeatTestRunner(BaseTestRunner):
    """Test runner for repeat job tests."""

    def __init__(self):
        super().__init__(
            prog_name="repeat_test.py",
            description="Run multiple repeats of a single job configuration",
            default_output_file="repeat_test_jobs.json",
            default_base_name="repeat_test",
            test_type="Repeat Test",
        )

    def get_launch_description(self) -> str:
        """Get the launch subcommand description."""
        return f"Launch {NUM_REPEATS} repeat jobs with the same configuration"

    def get_launch_epilog(self) -> str:
        """Get the launch subcommand epilog with examples."""
        return f"""
This launches {NUM_REPEATS} identical jobs with unique run names.

Examples:
  %(prog)s                      # Launch {NUM_REPEATS} repeat jobs
  %(prog)s --base-name mytest   # Launch with custom base name
  %(prog)s --skip-git-check     # Launch without git validation
        """

    def launch_tests(self, args):
        """Launch repeat test jobs."""
        # Create launcher
        launcher = SkyPilotTestLauncher(base_name=args.base_name, skip_git_check=args.skip_git_check)

        # Check git state
        if not launcher.check_git_state():
            sys.exit(1)

        # Show test configuration
        print(f"\n{bold('=== Skypilot Repeat Test Configuration ===')}")
        print(f"{cyan('Module:')} {BASE_MODULE}")
        print(f"{cyan('Base args:')} {' '.join(BASE_ARGS)}")
        print(f"{cyan('Run prefix:')} {RUN_PREFIX}")
        print(f"{cyan('Number of repeats:')} {NUM_REPEATS}")
        print(f"{cyan('Output file:')} {args.output_file}")

        # Launch jobs
        for i in range(1, NUM_REPEATS + 1):
            # Create run name with repeat number
            run_name = f"{RUN_PREFIX}.{i}"

            # Test config for tracking
            test_config = {
                "repeat_number": i,
                "total_repeats": NUM_REPEATS,
                "module": BASE_MODULE,
            }

            # Launch the job
            print(f"\n{yellow(f'Launching repeat {i}/{NUM_REPEATS}:')}")
            launcher.launch_job(
                module=BASE_MODULE,
                run_name=run_name,
                base_args=BASE_ARGS,
                extra_args=[],  # No extra args for repeats
                test_config=test_config,
                enable_ci_tests=False,
            )

        # Save results
        output_path = launcher.save_results(args.output_file)
        print(f"\n{cyan('Results saved to:')} {output_path.absolute()}")

        # Print summary
        launcher.print_summary()

        # Exit with error if any launches failed
        if launcher.failed_launches:
            sys.exit(1)


def main():
    """Main entry point."""
    runner = RepeatTestRunner()
    runner.run()


if __name__ == "__main__":
    main()
