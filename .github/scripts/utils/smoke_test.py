"""
Base class for smoke tests with benchmarking.

Provides common functionality for running commands, measuring performance,
and reporting results.
"""

import time
from abc import ABC, abstractmethod
from typing import Tuple

from .benchmark import run_with_benchmark, write_github_output


class SmokeTest(ABC):
    """Base class for all smoke tests."""

    def __init__(self) -> None:
        self.test_name = self.__class__.__name__.replace("SmokeTest", "").lower()
        self.timeout = self.get_timeout()

    @abstractmethod
    def get_command(self) -> list[str]:
        """Return the command to run for this test."""
        pass

    @abstractmethod
    def get_timeout(self) -> int:
        """Return the timeout for this test in seconds."""
        pass

    def print_header(self) -> None:
        """Print the test header."""
        print("=" * 60)
        print(f"{self.test_name.title()} Smoke Test")
        print("=" * 60)
        print(f"Timeout: {self.timeout}s")
        for line in self.header_config_lines():
            print(line)
        print("=" * 60)

    def header_config_lines(self) -> list[str]:
        """Override to print additional configuration."""
        return []

    def process_result(self, result: dict) -> Tuple[bool, str]:
        """
        Process the benchmark result.

        Args:
            result: The result from run_with_benchmark

        Returns:
            Tuple of (success, full_output)
        """
        if not result["success"]:
            print(f"{self.test_name.title()} failed with exit code: {result['exit_code']}")
            if result["timeout"]:
                print(f"{self.test_name.title()} timed out")
            elif result["stderr"]:
                print("STDERR output:")
                # Print full stderr without truncation
                print(result["stderr"])

                # Try to extract and highlight the actual error from the traceback
                stderr_lines = result["stderr"].split("\n")

                # Look for the actual exception at the end of the traceback
                for i in range(len(stderr_lines) - 1, max(0, len(stderr_lines) - 50), -1):
                    line = stderr_lines[i].strip()
                    # Check if this line contains an exception
                    if line and any(
                        exc in line
                        for exc in [
                            "Error",
                            "Exception",
                            "Error:",
                            "Exception:",
                            "AssertionError",
                            "ValueError",
                            "RuntimeError",
                            "TypeError",
                            "AttributeError",
                            "KeyError",
                            "IndexError",
                        ]
                    ):
                        print("\n" + "=" * 60)
                        print("Actual error (extracted from traceback):")
                        print("=" * 60)
                        # Print this line and a few lines of context
                        start_idx = max(0, i - 3)
                        end_idx = min(len(stderr_lines), i + 3)
                        for j in range(start_idx, end_idx):
                            if stderr_lines[j].strip():
                                print(stderr_lines[j])
                        print("=" * 60 + "\n")
                        break

        full_output = result["stdout"] + "\n" + result["stderr"]
        return result["success"], full_output

    def run_test(self) -> Tuple[bool, str, float, float]:
        """
        Run the test with benchmarking.

        Returns:
            Tuple of (success, full_output, duration, peak_memory_mb)
        """
        cmd = self.get_command()
        print(f"\nRunning {self.test_name}...")

        result = run_with_benchmark(cmd=cmd, name=self.test_name, timeout=self.timeout)

        success, full_output = self.process_result(result)
        return success, full_output, result["duration"], result["memory_peak_mb"]

    def print_summary(self, total_duration: float, test_duration: float, memory: float, success: bool) -> None:
        """Print the benchmark summary."""
        print(f"\n{'=' * 60}")
        print("Benchmark Summary")
        print(f"{'=' * 60}")
        print(f"Total duration: {total_duration:.1f}s")
        print(f"{self.test_name.title()} duration: {test_duration:.1f}s")
        print(f"Peak memory usage: {memory:.1f} MB")
        print(f"Exit status: {'SUCCESS' if success else 'FAILED'}")

    def write_outputs(self, total_duration: float, memory: float, success: bool, **extra_outputs) -> None:
        """Write GitHub Actions outputs."""
        outputs = {
            "duration": f"{total_duration:.1f}",
            "memory_peak_mb": f"{memory:.1f}",
            "exit_code": "0" if success else "1",
        }
        outputs.update(extra_outputs)
        write_github_output(outputs)

    def run(self) -> int:
        """Main entry point for the smoke test."""
        self.print_header()

        start_time = time.time()
        success, output, duration, memory = self.run_test()
        total_duration = time.time() - start_time

        self.print_summary(total_duration, duration, memory, success)
        self.write_outputs(total_duration, memory, success)

        return 0 if success else 1
