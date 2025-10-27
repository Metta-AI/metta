#!/usr/bin/env python3
"""Monitor comprehensive evaluation progress and populate results when complete.

This script:
1. Monitors the evaluation process (shell 3fa80c)
2. Periodically checks progress by reading the log file
3. When evaluation completes, analyzes results and updates the summary document
"""

import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path


def check_shell_status(shell_id: str) -> str:
    """Check if background shell is still running."""
    try:
        # Use ps to check if process is running
        result = subprocess.run(
            ["ps", "-p", shell_id],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return "running" if result.returncode == 0 else "completed"
    except subprocess.TimeoutExpired:
        return "unknown"
    except Exception as e:
        print(f"Error checking shell status: {e}")
        return "unknown"


def count_completed_tests(log_file: Path) -> tuple[int, int]:
    """Count how many tests have completed by scanning log file.

    Returns (completed, total_expected)
    """
    total_expected = 684  # 19 experiments × 4 difficulties × 9 presets

    if not log_file.exists():
        return 0, total_expected

    try:
        # Count lines with test completion markers
        completed = 0
        with open(log_file, "r") as f:
            for line in f:
                # Look for test completion patterns
                if "Test complete:" in line or "Result:" in line:
                    completed += 1

        return completed, total_expected
    except Exception as e:
        print(f"Error reading log file: {e}")
        return 0, total_expected


def analyze_results(results_file: Path) -> dict:
    """Analyze evaluation results when complete."""
    if not results_file.exists():
        return {}

    try:
        with open(results_file, "r") as f:
            results = json.load(f)
    except Exception as e:
        print(f"Error loading results: {e}")
        return {}

    # Analyze by difficulty
    by_difficulty = defaultdict(lambda: {"total": 0, "success": 0})
    by_preset = defaultdict(lambda: {"total": 0, "success": 0})
    by_experiment = defaultdict(lambda: {"total": 0, "success": 0})

    for result in results:
        # Extract difficulty from test name (e.g., "EXP1_easy_conservative")
        parts = result["test_name"].split("_")
        if len(parts) >= 2:
            difficulty = parts[1]  # easy, medium, hard, extreme

            by_difficulty[difficulty]["total"] += 1
            if result.get("success", False):
                by_difficulty[difficulty]["success"] += 1

        # By preset
        preset = result.get("preset", "unknown")
        by_preset[preset]["total"] += 1
        if result.get("success", False):
            by_preset[preset]["success"] += 1

        # By experiment
        experiment = parts[0] if parts else "unknown"
        by_experiment[experiment]["total"] += 1
        if result.get("success", False):
            by_experiment[experiment]["success"] += 1

    # Calculate percentages
    analysis = {
        "total_tests": len(results),
        "total_success": sum(1 for r in results if r.get("success", False)),
        "by_difficulty": {},
        "by_preset": {},
        "by_experiment": {},
    }

    for difficulty, stats in by_difficulty.items():
        analysis["by_difficulty"][difficulty] = {
            "total": stats["total"],
            "success": stats["success"],
            "rate": stats["success"] / stats["total"] if stats["total"] > 0 else 0,
        }

    for preset, stats in by_preset.items():
        analysis["by_preset"][preset] = {
            "total": stats["total"],
            "success": stats["success"],
            "rate": stats["success"] / stats["total"] if stats["total"] > 0 else 0,
        }

    for experiment, stats in by_experiment.items():
        analysis["by_experiment"][experiment] = {
            "total": stats["total"],
            "success": stats["success"],
            "rate": stats["success"] / stats["total"] if stats["total"] > 0 else 0,
        }

    return analysis


def update_summary_document(analysis: dict, output_file: Path):
    """Update the performance summary document with actual results."""

    # Create summary markdown
    summary = f"""# Comprehensive Evaluation Results

**Evaluation completed**: {time.strftime("%Y-%m-%d %H:%M:%S")}
**Total tests**: {analysis['total_tests']}
**Overall success rate**: {analysis['total_success'] / analysis['total_tests'] * 100:.1f}%

## Results by Difficulty

| Difficulty | Tests | Success | Rate |
|------------|-------|---------|------|
"""

    for difficulty in ["easy", "medium", "hard", "extreme"]:
        if difficulty in analysis["by_difficulty"]:
            stats = analysis["by_difficulty"][difficulty]
            summary += f"| {difficulty.upper()} | {stats['total']} | {stats['success']} | {stats['rate']*100:.1f}% |\n"

    summary += "\n## Results by Hyperparameter Preset\n\n| Preset | Tests | Success | Rate |\n|--------|-------|---------|------|\n"

    # Sort presets by success rate
    sorted_presets = sorted(
        analysis["by_preset"].items(),
        key=lambda x: x[1]["rate"],
        reverse=True
    )

    for preset, stats in sorted_presets:
        summary += f"| {preset} | {stats['total']} | {stats['success']} | {stats['rate']*100:.1f}% |\n"

    summary += "\n## Results by Experiment\n\n| Experiment | Tests | Success | Rate |\n|------------|-------|---------|------|\n"

    # Sort experiments by success rate
    sorted_experiments = sorted(
        analysis["by_experiment"].items(),
        key=lambda x: x[1]["rate"],
        reverse=True
    )

    for experiment, stats in sorted_experiments:
        summary += f"| {experiment} | {stats['total']} | {stats['success']} | {stats['rate']*100:.1f}% |\n"

    # Write to file
    with open(output_file, "w") as f:
        f.write(summary)

    print(f"\nSummary written to: {output_file}")


def main():
    """Main monitoring loop."""

    shell_id = "3fa80c"
    log_file = Path("/Users/daphnedemekas/Desktop/metta/difficulty_evaluation_comprehensive.log")
    results_file = Path("/Users/daphnedemekas/Desktop/metta/difficulty_results_comprehensive.json")
    summary_file = Path("/Users/daphnedemekas/Desktop/metta/EVALUATION_RESULTS_FINAL.md")

    print("=" * 80)
    print("COMPREHENSIVE EVALUATION MONITOR")
    print("=" * 80)
    print(f"Shell ID: {shell_id}")
    print(f"Log file: {log_file}")
    print(f"Results file: {results_file}")
    print(f"Will update: {summary_file}")
    print("=" * 80)
    print()

    check_interval = 300  # Check every 5 minutes
    last_completed = 0

    while True:
        # Check if process is still running
        status = check_shell_status(shell_id)

        # Count completed tests
        completed, total = count_completed_tests(log_file)

        # Show progress if changed
        if completed != last_completed:
            progress_pct = completed / total * 100 if total > 0 else 0
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Progress: {completed}/{total} tests ({progress_pct:.1f}%)")
            last_completed = completed

        # If completed, analyze and exit
        if status == "completed" or completed >= total:
            print("\n" + "=" * 80)
            print("EVALUATION COMPLETE!")
            print("=" * 80)

            # Wait a moment for file writes to complete
            time.sleep(5)

            # Analyze results
            print("\nAnalyzing results...")
            analysis = analyze_results(results_file)

            if analysis:
                print(f"Total tests: {analysis['total_tests']}")
                print(f"Overall success: {analysis['total_success']} / {analysis['total_tests']} ({analysis['total_success'] / analysis['total_tests'] * 100:.1f}%)")
                print()

                # Update summary document
                update_summary_document(analysis, summary_file)

                print("\nResults summary:")
                print(f"- Easy: {analysis['by_difficulty'].get('easy', {}).get('rate', 0)*100:.1f}%")
                print(f"- Medium: {analysis['by_difficulty'].get('medium', {}).get('rate', 0)*100:.1f}%")
                print(f"- Hard: {analysis['by_difficulty'].get('hard', {}).get('rate', 0)*100:.1f}%")
                print(f"- Extreme: {analysis['by_difficulty'].get('extreme', {}).get('rate', 0)*100:.1f}%")

                print("\nTop 3 presets:")
                sorted_presets = sorted(
                    analysis["by_preset"].items(),
                    key=lambda x: x[1]["rate"],
                    reverse=True
                )
                for preset, stats in sorted_presets[:3]:
                    print(f"  {preset}: {stats['rate']*100:.1f}%")
            else:
                print("No results to analyze yet.")

            print("\n" + "=" * 80)
            print("Monitoring complete.")
            sys.exit(0)

        # Wait before next check
        time.sleep(check_interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nMonitoring interrupted by user.")
        sys.exit(0)
