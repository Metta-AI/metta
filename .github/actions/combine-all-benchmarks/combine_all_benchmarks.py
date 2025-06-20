#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
Combine Python and C++ benchmark results into unified bencher BMF format.
"""

import glob
import json
import os


def safe_load_json(file_path):
    try:
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                content = f.read().strip()
                if content:
                    return json.loads(content)
        return {}
    except (json.JSONDecodeError, FileNotFoundError):
        print(f"Warning: Could not load {file_path}")
        return {}


def convert_pytest_to_bmf_format(pytest_data):
    """Convert pytest-benchmark to BMF using only 2 KPIs (no latency)"""
    bmf_results = {}

    for bench in pytest_data.get("benchmarks", []):
        name = bench.get("fullname", bench.get("name", "unknown"))
        extra_info = bench.get("extra_info", {})

        bench_metrics = {}

        kpis = ["agent_rate", "env_rate"]

        for kpi_name in kpis:
            if kpi_name in extra_info and isinstance(extra_info[kpi_name], (int, float)):
                bench_metrics[kpi_name] = {"value": extra_info[kpi_name]}

        if bench_metrics:
            bmf_results[name] = bench_metrics

    return bmf_results


def convert_cpp_to_bmf_format(cpp_data):
    """Convert Google Benchmark to BMF using only custom counters (no latency)"""
    bmf_results = {}

    for bench in cpp_data.get("benchmarks", []):
        name = bench.get("name", "unknown")
        bench_metrics = {}

        # Debug: print all keys in the benchmark to understand structure
        print(f"Benchmark '{name}' has keys: {list(bench.keys())}")

        # Check if counters are in user_counters or directly in bench
        counters = bench.get("user_counters", bench)

        # Extract custom counters (agent_rate, env_rate)
        kpis = ["agent_rate", "env_rate"]

        for kpi_name in kpis:
            if kpi_name in counters:
                value = counters[kpi_name]
                # Handle case where value might be a dict with 'value' key
                if isinstance(value, dict) and "value" in value:
                    value = value["value"]
                if isinstance(value, (int, float)):
                    # Check if the value seems unreasonably small (indicating it might be in different units)
                    if value < 1000 and kpi_name.endswith("_per_second"):
                        print(f"WARNING: {kpi_name} value {value} seems too small, might be scaled")
                    bench_metrics[kpi_name] = {"value": value}
                else:
                    print(f"Skipping {kpi_name} with non-numeric value: {value}")

        if bench_metrics:
            bmf_results[name] = bench_metrics
        else:
            print(f"No valid metrics found for benchmark '{name}'")

    return bmf_results


def main():
    """Main entry point."""
    # Get inputs from environment variables
    python_files = os.environ.get("PYTHON_FILES", "").split(",")
    cpp_pattern = os.environ.get("CPP_FILES", "")
    output_file = os.environ.get("OUTPUT_FILE", "unified_benchmark_results.json")

    # Initialize unified results in BMF format
    unified_results = {}

    # Process Python benchmark files
    for file_path in python_files:
        file_path = file_path.strip()
        if not file_path:
            continue

        data = safe_load_json(file_path)
        if data and "benchmarks" in data and len(data["benchmarks"]) > 0:
            print(f"Processing Python file: {file_path} ({len(data['benchmarks'])} benchmarks)")
            bmf_results = convert_pytest_to_bmf_format(data)

            # Add source prefix to benchmark names
            for bench_name, bench_metrics in bmf_results.items():
                prefixed_name = f"python/{bench_name}"
                unified_results[prefixed_name] = bench_metrics

            print(f"Added {len(bmf_results)} Python benchmarks")
        else:
            print(f"Skipping empty or invalid Python file: {file_path}")

    # Process C++ benchmark files
    cpp_files = glob.glob(cpp_pattern)

    for file_path in cpp_files:
        # Skip if it's actually a Python file that got picked up
        if any(py_file.strip().endswith(os.path.basename(file_path)) for py_file in python_files):
            continue

        data = safe_load_json(file_path)
        if data and "benchmarks" in data and len(data["benchmarks"]) > 0:
            print(f"\nProcessing C++ file: {file_path} ({len(data['benchmarks'])} benchmarks)")

            # Debug: print full structure of first benchmark
            sample_bench = data["benchmarks"][0]
            print("\nSample C++ benchmark FULL structure:")
            print(json.dumps(sample_bench, indent=2))

            bmf_results = convert_cpp_to_bmf_format(data)

            # Add source prefix to benchmark names
            for bench_name, bench_metrics in bmf_results.items():
                prefixed_name = f"cpp/{bench_name}"
                unified_results[prefixed_name] = bench_metrics

            print(f"Added {len(bmf_results)} C++ benchmarks")
        else:
            print(f"Skipping empty or invalid C++ file: {file_path}")

    # Write unified results in BMF format
    if len(unified_results) == 0:
        print("⚠️ Warning: No benchmarks found to combine!")
        # Create empty BMF object
        with open(output_file, "w") as f:
            json.dump({}, f)
    else:
        with open(output_file, "w") as f:
            json.dump(unified_results, f, indent=2)

    print(f"\n✅ Combined {len(unified_results)} benchmarks in BMF format")

    # Debug: Print sample of what we generated for both Python and C++
    if unified_results:
        print("\n📋 Sample BMF outputs:")
        # Show one Python and one C++ example if available
        for prefix in ["python/", "cpp/"]:
            for key in unified_results:
                if key.startswith(prefix):
                    print(f"\n{key}:")
                    print(json.dumps(unified_results[key], indent=2))
                    break


if __name__ == "__main__":
    main()
