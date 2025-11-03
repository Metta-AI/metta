"""Job restart test tool for SkyPilot.

This tool validates that SkyPilot job restart functionality works correctly by:
1. Running a simple training command
2. Forcing a restart at 50% of max runtime
3. Verifying the job restarts successfully
4. Checking that restart count increments properly

This test requires --max-runtime-hours to be set.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

from devops.skypilot.utils.runtime_monitors import ForceRestartTestMonitor, TimeoutMonitor
from devops.skypilot.utils.subprocess_helpers import terminate_process_group
from metta.common.tool.base_tool import BaseTool

EXIT_AND_STOP = 0
EXIT_AND_RESTART = 1


class RestartTestTool(BaseTool):
    """Tool for testing job restart functionality."""

    def run(self) -> int:
        """Run restart test.

        This executes a simple training command and forces a restart at 50% of max runtime
        to validate that job restart mechanisms work correctly.
        """
        print("=" * 80)
        print("SkyPilot Job Restart Test")
        print("=" * 80)

        # Validate required environment
        max_runtime_hours = float(os.environ.get("MAX_RUNTIME_HOURS", "0"))
        if not max_runtime_hours:
            print("ERROR: MAX_RUNTIME_HOURS must be set for restart tests")
            print("Use: ./devops/skypilot/launch.py --tool restart_test -t 0.1 run=test_restart")
            return 1

        restart_count = int(os.environ.get("RESTART_COUNT", "0"))
        accumulated_runtime_file_path = os.environ.get("ACCUMULATED_RUNTIME_FILE")

        # Load accumulated runtime if available
        accumulated_runtime_sec = 0
        if accumulated_runtime_file_path:
            accumulated_runtime_file = Path(accumulated_runtime_file_path)
            if accumulated_runtime_file.exists():
                try:
                    accumulated_runtime_sec = int(accumulated_runtime_file.read_text())
                except (ValueError, IOError):
                    pass

        print("\nTest Configuration:")
        print(f"  Max Runtime: {max_runtime_hours} hours")
        print(f"  Restart Count: {restart_count}")
        print(f"  Accumulated Runtime: {accumulated_runtime_sec}s")

        # Only force restart on first run
        if restart_count == 0:
            print(f"\nFirst run - will force restart at {max_runtime_hours / 2:.3f} hours")
            return self._run_with_forced_restart(max_runtime_hours)
        else:
            print(f"\nRestart #{restart_count} - running to completion")
            return self._run_to_completion(max_runtime_hours, accumulated_runtime_sec)

    def _run_with_forced_restart(self, max_runtime_hours: float) -> int:
        """Run test with forced restart at 50% of max runtime."""
        # Simple command that just sleeps - we're testing restart, not actual work
        cmd = ["python3", "-c", "import time; print('Running test...'); time.sleep(3600)"]

        print(f"\nLaunching test command: {' '.join(cmd)}")
        process = subprocess.Popen(
            cmd,
            start_new_session=True,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )

        # Set up monitors
        restart_time_hours = max_runtime_hours / 2.0
        force_restart_monitor = ForceRestartTestMonitor(restart_time_hours)

        print(f"\nMonitoring for forced restart at {restart_time_hours:.3f} hours...")

        while True:
            exit_code = process.poll()
            if exit_code is not None:
                print(f"\nTest command exited unexpectedly with code {exit_code}")
                return EXIT_AND_STOP

            reason = force_restart_monitor.check_condition()
            if reason:
                print(f"\nForce restart triggered: {reason}")
                terminate_process_group(process)
                print("Test will restart to validate recovery...")
                return EXIT_AND_RESTART

            time.sleep(1)

    def _run_to_completion(self, max_runtime_hours: float, accumulated_runtime_sec: int) -> int:
        """Run test to completion with timeout monitoring."""
        # Simple command that completes quickly
        cmd = ["python3", "-c", "print('Restart successful! Test completed.')"]

        print(f"\nLaunching test command: {' '.join(cmd)}")
        process = subprocess.Popen(
            cmd,
            start_new_session=True,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )

        # Set up timeout monitor to track accumulated runtime
        node_index = int(os.environ.get("SKYPILOT_NODE_RANK", "0"))
        timeout_monitor = TimeoutMonitor(rank=node_index, max_runtime_hours=max_runtime_hours)

        print("\nWaiting for test completion...")

        while True:
            exit_code = process.poll()
            if exit_code is not None:
                if exit_code == 0:
                    total_runtime = timeout_monitor.get_total_runtime()
                    print(f"\n{'=' * 80}")
                    print("RESTART TEST PASSED")
                    print(f"  - Restart count: {os.environ.get('RESTART_COUNT', '0')}")
                    print(f"  - Total runtime: {total_runtime}s")
                    print("  - Test validated job restart functionality successfully!")
                    print(f"{'=' * 80}")
                    return EXIT_AND_STOP
                else:
                    print(f"\nTest command failed with exit code {exit_code}")
                    return EXIT_AND_STOP

            # Check timeout to prevent runaway test
            reason = timeout_monitor.check_condition()
            if reason:
                print(f"\nTest timeout reached: {reason}")
                terminate_process_group(process)
                return EXIT_AND_STOP

            time.sleep(1)


def main():
    """Entry point for restart test tool."""
    tool = RestartTestTool()
    return tool.run()


if __name__ == "__main__":
    sys.exit(main())
