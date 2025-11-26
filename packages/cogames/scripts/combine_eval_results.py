#!/usr/bin/env -S uv run
"""
Combine multiple evaluation JSON files and regenerate plots.
"""

import json
import sys
from pathlib import Path
from typing import List

# Add the scripts directory to path for imports
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

# Import directly from the same directory
from run_evaluation import EvalResult, create_plots


def combine_eval_results(json_files: List[str], output_json: str, plot_dir: str = "eval_plots") -> None:
    """Combine multiple evaluation JSON files and regenerate plots."""
    all_results: List[EvalResult] = []

    for json_file in json_files:
        json_path = Path(json_file)
        if not json_path.exists():
            print(f"Warning: {json_file} not found, skipping...")
            continue

        print(f"Loading results from {json_file}...")
        with open(json_path, "r") as f:
            results_data = json.load(f)

        # Convert dicts to EvalResult objects
        results = [EvalResult(**r) for r in results_data]
        all_results.extend(results)
        print(f"  Loaded {len(results)} results")

    print(f"\nTotal results: {len(all_results)}")

    # Save combined results
    if output_json:
        output_path = Path(output_json)
        with open(output_path, "w") as f:
            json.dump([r.__dict__ for r in all_results], f, indent=2)
        print(f"✓ Combined results saved to {output_json}")

    # Generate plots
    if all_results:
        print(f"\nGenerating plots in {plot_dir}/...")
        create_plots(all_results, output_dir=plot_dir)
        print(f"✓ Plots saved to {plot_dir}/")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: combine_eval_results.py <json1> <json2> [json3...] --output <output.json> [--plot-dir <dir>]")
        print("Example: combine_eval_results.py go_together_all_variants.json full_curriculum_no_diagnostics_v2_eval_results.json --output combined_results.json")
        sys.exit(1)

    json_files = []
    output_json = None
    plot_dir = "eval_plots"

    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--output" and i + 1 < len(sys.argv):
            output_json = sys.argv[i + 1]
            i += 2
        elif arg == "--plot-dir" and i + 1 < len(sys.argv):
            plot_dir = sys.argv[i + 1]
            i += 2
        elif not arg.startswith("--"):
            json_files.append(arg)
            i += 1
        else:
            i += 1

    if not json_files:
        print("Error: No JSON files specified")
        sys.exit(1)

    combine_eval_results(json_files, output_json, plot_dir)

