#!/usr/bin/env python3
"""
Script to compare navigation training with and without exploration tracking.

This script helps you run both versions and provides guidance on comparing the results.
"""

import subprocess
import sys


def run_command(cmd, description):
    """Run a command and return the result."""
    print(f"\n{'=' * 60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'=' * 60}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Command completed successfully")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed with exit code {e.returncode}")
        print(f"Error output: {e.stderr}")
        return None


def main():
    """Main function to run navigation comparison."""
    print("Navigation Training Comparison Tool")
    print("==================================")
    print()
    print("This tool helps you compare navigation training with and without exploration tracking.")
    print()
    print("Options:")
    print("1. Run baseline navigation training (no exploration tracking)")
    print("2. Run navigation training with exploration tracking")
    print("3. Run both and compare")
    print("4. Show comparison guidance")
    print()

    choice = input("Enter your choice (1-4): ").strip()

    if choice == "1":
        print("\nRunning baseline navigation training...")
        cmd = ["./recipes/navigation.sh"]
        run_command(cmd, "Baseline Navigation Training")

    elif choice == "2":
        print("\nRunning navigation training with exploration tracking...")
        cmd = ["./recipes/navigation_exploration.sh"]
        run_command(cmd, "Navigation Training with Exploration Tracking")

    elif choice == "3":
        print("\nRunning both versions for comparison...")

        # Run baseline
        baseline_result = run_command(["./recipes/navigation.sh"], "Baseline Navigation Training")

        # Run with exploration
        exploration_result = run_command(
            ["./recipes/navigation_exploration.sh"], "Navigation Training with Exploration Tracking"
        )

        if baseline_result and exploration_result:
            print("\n✅ Both runs completed successfully!")
            print("\nTo compare results:")
            print("1. Check the wandb runs for both experiments")
            print("2. Look for 'exploration_rate' metrics in the exploration-enabled run")
            print("3. Compare overall performance metrics between runs")

    elif choice == "4":
        print("\nComparison Guidance:")
        print("====================")
        print()
        print("1. Run both versions:")
        print("   ./recipes/navigation.sh")
        print("   ./recipes/navigation_exploration.sh")
        print()
        print("2. In the exploration-enabled run, look for:")
        print("   - 'exploration_rate' in agent episode stats")
        print("   - This metric shows unique pixels explored per step")
        print()
        print("3. Compare metrics:")
        print("   - Overall success rates")
        print("   - Episode completion times")
        print("   - Exploration rates vs performance")
        print()
        print("4. Key questions to investigate:")
        print("   - Do agents with higher exploration rates perform better?")
        print("   - Is there a correlation between exploration and navigation success?")
        print("   - How does exploration rate vary across different navigation tasks?")
        print()
        print("5. Wandb integration:")
        print("   - Exploration rates are automatically logged to wandb")
        print("   - You can create custom plots comparing exploration vs performance")
        print("   - Filter by evaluation environment to see task-specific patterns")

    else:
        print("Invalid choice. Please enter 1, 2, 3, or 4.")
        sys.exit(1)


if __name__ == "__main__":
    main()
