#!/usr/bin/env python
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
Test script to verify that all MettaGrid demo environments can start and run without crashing.

This script runs each demo for 60 seconds to ensure they're functional and don't crash.
"""

import signal
import subprocess
import sys
import time
from pathlib import Path


class TimeoutError(Exception):
    """Custom timeout exception."""

    pass


def timeout_handler(signum, frame):
    """Handle timeout signal."""
    raise TimeoutError("Demo timed out")


def test_demo(demo_path: Path, timeout_seconds: int = 60) -> bool:
    """
    Test a single demo by running it for the specified timeout period.

    Args:
        demo_path: Path to the demo script
        timeout_seconds: Maximum time to run the demo

    Returns:
        True if demo runs successfully, False otherwise
    """
    print(f"Testing {demo_path.name}...")

    try:
        # Start the demo process
        process = subprocess.Popen(
            ["uv", "run", "python", str(demo_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_IGN),
        )

        # Wait for the specified timeout
        start_time = time.time()
        while time.time() - start_time < timeout_seconds:
            if process.poll() is not None:
                # Process has terminated
                stdout, stderr = process.communicate()
                if process.returncode == 0:
                    print(f"✅ {demo_path.name} completed successfully")
                    return True
                else:
                    print(f"❌ {demo_path.name} failed with return code {process.returncode}")
                    if stderr:
                        print(f"Error output: {stderr}")
                    return False
            time.sleep(0.1)

        # If we get here, the demo ran for the full timeout period
        print(f"✅ {demo_path.name} ran successfully for {timeout_seconds} seconds")

        # Terminate the process
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

        return True

    except Exception as e:
        print(f"❌ {demo_path.name} failed with exception: {e}")
        return False


def main():
    """Run all demo tests."""
    print("Testing MettaGrid Demo Environments")
    print("=" * 40)

    # Find all demo scripts
    demo_dir = Path(__file__).parent
    demo_scripts = [
        demo_dir / "gym_demo.py",
        demo_dir / "pettingzoo_demo.py",
        demo_dir / "puffer_demo.py",
    ]

    # Test each demo
    results = {}
    for demo_path in demo_scripts:
        if demo_path.exists():
            results[demo_path.name] = test_demo(demo_path, timeout_seconds=60)
        else:
            print(f"❌ {demo_path.name} not found")
            results[demo_path.name] = False

    # Print summary
    print("\n" + "=" * 40)
    print("Test Summary:")
    print("-" * 40)

    all_passed = True
    for demo_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{demo_name:<25} {status}")
        if not passed:
            all_passed = False

    print("-" * 40)
    if all_passed:
        print("🎉 All demos passed!")
        sys.exit(0)
    else:
        print("💥 Some demos failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
