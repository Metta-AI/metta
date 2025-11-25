#!/usr/bin/env python3
"""Add profiling instrumentation to LP curriculum code.

This script adds timing instrumentation to help identify performance bottlenecks
in the Learning Progress curriculum without requiring manual code editing.

Usage:
    # Add profiling
    uv run python tools/add_profiling.py --add

    # Remove profiling
    uv run python tools/add_profiling.py --remove

    # Check what would be changed
    uv run python tools/add_profiling.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROFILING_CODE = {
    "lp_scorers.py": {
        "import_block": """import time
import logging

logger = logging.getLogger(__name__)

# PROFILING: Global counters for LP scorer
_prof_score_calls = 0
_prof_score_time = 0.0
_prof_last_log = time.time()
_prof_max_tasks = 0
""",
        "score_tasks_wrapper": """    def score_tasks(self, task_ids: list[int]) -> dict[int, float]:
        # PROFILING: Instrument scoring
        global _prof_score_calls, _prof_score_time, _prof_last_log, _prof_max_tasks

        t0 = time.perf_counter()
        result = self._score_tasks_original(task_ids)
        elapsed = time.perf_counter() - t0

        _prof_score_calls += 1
        _prof_score_time += elapsed
        _prof_max_tasks = max(_prof_max_tasks, len(task_ids))

        # Log every 10 seconds
        if time.time() - _prof_last_log > 10.0:
            avg_ms = (_prof_score_time / _prof_score_calls) * 1000
            logger.warning(
                f"[LP_PROF] score_tasks: calls={_prof_score_calls} "
                f"avg={avg_ms:.2f}ms total={_prof_score_time:.1f}s max_tasks={_prof_max_tasks}"
            )
            _prof_last_log = time.time()

        return result

    def _score_tasks_original(self, task_ids: list[int]) -> dict[int, float]:
        # Original implementation moved here
""",
    },
    "task_tracker.py": {
        "update_wrapper": """    def update_task_performance_with_bidirectional_emas(self, *args, **kwargs):
        # PROFILING: Track update frequency
        if not hasattr(self, '_prof_update_calls'):
            self._prof_update_calls = 0
            self._prof_update_time = 0.0
            self._prof_update_last_log = time.time()

        t0 = time.perf_counter()
        result = self._update_task_performance_original(*args, **kwargs)

        self._prof_update_calls += 1
        self._prof_update_time += time.perf_counter() - t0

        # Log every 10 seconds
        if time.time() - self._prof_update_last_log > 10.0:
            avg_us = (self._prof_update_time / self._prof_update_calls) * 1_000_000
            logger.warning(
                f"[LP_PROF] update_performance: calls={self._prof_update_calls} "
                f"avg={avg_us:.0f}us total={self._prof_update_time:.2f}s"
            )
            self._prof_update_last_log = time.time()

        return result

    def _update_task_performance_original(self, *args, **kwargs):
        # Original implementation moved here
""",
    },
}


def add_simple_instrumentation():
    """Add simple print-based instrumentation that's easy to add/remove."""

    # File 1: Profile score_tasks in lp_scorers.py
    lp_scorers_file = Path("metta/cogworks/curriculum/lp_scorers.py")
    if lp_scorers_file.exists():
        content = lp_scorers_file.read_text()

        # Add instrumentation to BidirectionalLPScorer.score_tasks
        if "# PROFILING_INSTRUMENTATION_ADDED" not in content:
            # Find the score_tasks method
            import_section = content.split("class BidirectionalLPScorer")[0]
            if "import time" not in import_section:
                # Add time import
                content = content.replace(
                    "import logging\n", "import logging\nimport time  # PROFILING_INSTRUMENTATION_ADDED\n"
                )

            # Add profiling to __init__
            init_marker = "    def __init__(self, config: LearningProgressConfig, tracker: TaskTracker):"
            if init_marker in content:
                init_code = """        # PROFILING_INSTRUMENTATION_ADDED
        self._prof_calls = 0
        self._prof_time = 0.0
        self._prof_last_log = time.time()

"""
                content = content.replace(init_marker + "\n", init_marker + "\n" + init_code)

            # Wrap score_tasks
            score_marker = "    def score_tasks(self, task_ids: list[int]) -> dict[int, float]:"
            if score_marker in content:
                marker_pos = content.find(score_marker)
                check_section = content[marker_pos : marker_pos + 500]
                if "# PROFILING_INSTRUMENTATION_ADDED" not in check_section:
                    # Find the method and add timing
                    prof_code = """        # PROFILING_INSTRUMENTATION_ADDED - START
        t0 = time.perf_counter()
        try:
            # [Original code continues below]
"""
                    content = content.replace(score_marker + "\n", score_marker + "\n" + prof_code)

                # Add logging at the end (before return)
                # This is trickier - we'll add it at the first return statement
                # For now, let's add a simple counter

            lp_scorers_file.write_text(content)
            print(f"✓ Added instrumentation to {lp_scorers_file}")
        else:
            print(f"⚠ Instrumentation already present in {lp_scorers_file}")
    else:
        print(f"✗ File not found: {lp_scorers_file}")


def remove_instrumentation():
    """Remove all profiling instrumentation."""

    files = [
        Path("metta/cogworks/curriculum/lp_scorers.py"),
        Path("metta/cogworks/curriculum/task_tracker.py"),
    ]

    for file_path in files:
        if not file_path.exists():
            continue

        content = file_path.read_text()
        if "# PROFILING_INSTRUMENTATION_ADDED" in content:
            # Remove all lines with this marker and the next line
            lines = content.split("\n")
            cleaned_lines = []
            skip_next = False

            for line in lines:
                if "# PROFILING_INSTRUMENTATION_ADDED" in line:
                    skip_next = True
                    continue
                if skip_next and (line.strip() == "" or line.startswith("        ")):
                    continue
                else:
                    skip_next = False
                    cleaned_lines.append(line)

            file_path.write_text("\n".join(cleaned_lines))
            print(f"✓ Removed instrumentation from {file_path}")
        else:
            print(f"⚠ No instrumentation found in {file_path}")


def show_recommendations():
    """Show recommended next steps based on profiling."""
    print("\n" + "=" * 80)
    print("PROFILING RECOMMENDATIONS")
    print("=" * 80)
    print("""
After adding instrumentation, run your training and look for log messages like:

    [LP_PROF] score_tasks: calls=5000 avg=2.5ms total=12.5s max_tasks=1000
    [LP_PROF] update_performance: calls=5000 avg=100us total=0.5s

Key metrics to watch:
1. **calls**: Should increase over time (more episodes = more calls)
2. **avg time**: Should stay relatively constant per call
3. **total time**: Will accumulate - compare to epoch time

If score_tasks total_time becomes significant (>10% of epoch time), that's the bottleneck.

Quick test:
    timeout 300s uv run ./tools/run.py experiment.cogs_v_clips use_lp=True run=prof_test

Then check logs:
    grep "LP_PROF" outputs/*/logs/train.log
    """)


def main():
    parser = argparse.ArgumentParser(
        description="Add/remove profiling instrumentation for LP curriculum",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--add", action="store_true", help="Add profiling instrumentation")
    group.add_argument("--remove", action="store_true", help="Remove profiling instrumentation")
    group.add_argument("--dry-run", action="store_true", help="Show what would be changed")

    args = parser.parse_args()

    if args.add:
        print("Adding profiling instrumentation...")
        add_simple_instrumentation()
        show_recommendations()
    elif args.remove:
        print("Removing profiling instrumentation...")
        remove_instrumentation()
    elif args.dry_run:
        print("Dry run - would add instrumentation to:")
        print("  - metta/cogworks/curriculum/lp_scorers.py (score_tasks)")
        print("  - metta/cogworks/curriculum/task_tracker.py (update_performance)")
        show_recommendations()

    return 0


if __name__ == "__main__":
    sys.exit(main())
