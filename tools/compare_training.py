#!/usr/bin/env python3
"""
Compare Training Tool

This tool trains N pairs of policies:
- N policies using the functional trainer (bullm_run.py)
- N policies using the standard Hydra pipeline

Both use identical hyperparameters and compute instances for fair comparison.
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple

# Add the metta directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

# from omegaconf import OmegaConf  # Not used in this script

# Set up logging with a more specific format to avoid conflicts
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [COMPARE] %(levelname)s: %(message)s",
    force=True,  # Force reconfiguration to avoid conflicts
)
logger = logging.getLogger(__name__)

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("wandb not available - metrics comparison will be limited")


DEFAULT_HOURLY_RATE = 1.0  # USD per GPU hour (adjust as needed)


def run_functional_training(run_name: str, run_dir: str, group: str) -> Tuple[subprocess.Popen, str]:
    """Run functional training using bullm_run.py"""
    logger.info(f"Starting functional training: {run_name}")
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp_file:
        with open("bullm_run.py", "r") as f:
            content = f.read()
        tmp_file.write(content)
        tmp_file_path = tmp_file.name
    try:
        # Set environment variables for W&B project and group, and run name
        env = os.environ.copy()
        env["WANDB_PROJECT"] = "comparision_trainer"
        env["WANDB_GROUP"] = group
        env["RUN_NAME"] = run_name
        env["RUN_DIR"] = run_dir

        cmd = [sys.executable, tmp_file_path]
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=os.getcwd(), env=env
        )
        return process, tmp_file_path
    except Exception as e:
        os.unlink(tmp_file_path)
        raise e


def run_hydra_training(run_name: str, run_dir: str, group: str) -> subprocess.Popen:
    """Run Hydra training using the standard pipeline"""
    logger.info(f"Starting Hydra training: {run_name}")
    env = os.environ.copy()
    env["RUN_NAME"] = run_name
    env["RUN_DIR"] = run_dir
    env["WANDB_GROUP"] = group
    cmd = [
        sys.executable,
        "metta/tools/train.py",
        "--config-name",
        "user/compare_training",
        f"run={run_name}",
    ]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=os.getcwd(), env=env)
    return process


def monitor_process(process: subprocess.Popen, name: str) -> Tuple[int, str, str]:
    logger.info(f"Monitoring {name} training process...")

    stdout_lines = []
    stderr_lines = []

    while True:
        if process.stdout:
            stdout_line = process.stdout.readline()
            if stdout_line:
                stdout_lines.append(stdout_line.strip())
                logger.info(f"[{name}] {stdout_line.strip()}")

        if process.stderr:
            stderr_line = process.stderr.readline()
            if stderr_line:
                stderr_lines.append(stderr_line.strip())
                logger.warning(f"[{name}] {stderr_line.strip()}")

        if process.poll() is not None:
            break

        time.sleep(0.1)

    remaining_stdout, remaining_stderr = process.communicate()
    if remaining_stdout:
        stdout_lines.extend(remaining_stdout.strip().split("\n"))
    if remaining_stderr:
        stderr_lines.extend(remaining_stderr.strip().split("\n"))

    exit_code = process.returncode
    stdout_output = "\n".join(stdout_lines)
    stderr_output = "\n".join(stderr_lines)

    if exit_code == 0:
        logger.info(f"{name} training completed successfully")
    else:
        logger.error(f"{name} training failed with exit code {exit_code}")

    return exit_code, stdout_output, stderr_output


def fetch_wandb_metrics(
    run_name: str, group: str, project: str = "comparision_trainer", entity: str = "metta-research"
):
    """Fetch metrics from W&B for a given run name and group."""
    if not WANDB_AVAILABLE:
        logger.warning("W&B not available - cannot fetch metrics")
        return None

    try:
        api = wandb.Api()  # type: ignore
        runs = api.runs(f"{entity}/{project}", filters={"group": group, "name": run_name})
        if not runs:
            logger.warning(f"No W&B run found for {run_name} in group {group}")
            return None
        run = runs[0]
        summary = run.summary
        wall_time = (run.stop_time - run.created_at).total_seconds() if run.stop_time and run.created_at else None
        samples = summary.get("env_steps") or summary.get("samples") or summary.get("total_timesteps")
        hearts = summary.get("hearts.get")
        price = (wall_time / 3600) * DEFAULT_HOURLY_RATE if wall_time else None
        return {
            "run_name": run_name,
            "group": group,
            "wall_time": wall_time,
            "samples": samples,
            "hearts.get": hearts,
            "price": price,
            "url": run.url,
        }
    except Exception as e:
        logger.error(f"Error fetching W&B metrics for {run_name}: {e}")
        return None


def run_comparison_pair(pair_id: int, base_run_name: str, hourly_rate: float = DEFAULT_HOURLY_RATE) -> dict:
    logger.info(f"Starting comparison pair {pair_id}")
    group = f"compare_pair_{pair_id:02d}"
    functional_run_name = f"{base_run_name}_functional_{pair_id:02d}"
    hydra_run_name = f"{base_run_name}_hydra_{pair_id:02d}"
    functional_run_dir = f"train_dir/{functional_run_name}"
    hydra_run_dir = f"train_dir/{hydra_run_name}"
    results = {
        "pair_id": pair_id,
        "functional_run_name": functional_run_name,
        "hydra_run_name": hydra_run_name,
        "functional_exit_code": None,
        "hydra_exit_code": None,
        "functional_stdout": "",
        "functional_stderr": "",
        "hydra_stdout": "",
        "hydra_stderr": "",
        "functional_wandb": None,
        "hydra_wandb": None,
    }
    try:
        functional_process, tmp_file_path = run_functional_training(functional_run_name, functional_run_dir, group)
        hydra_process = run_hydra_training(hydra_run_name, hydra_run_dir, group)
        logger.info("Both training processes started. Monitoring...")
        results["functional_exit_code"], results["functional_stdout"], results["functional_stderr"] = monitor_process(
            functional_process, "Functional"
        )
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
        results["hydra_exit_code"], results["hydra_stdout"], results["hydra_stderr"] = monitor_process(
            hydra_process, "Hydra"
        )
        # Fetch W&B metrics
        results["functional_wandb"] = fetch_wandb_metrics(functional_run_name, group)
        results["hydra_wandb"] = fetch_wandb_metrics(hydra_run_name, group)
        # Print summary table
        print("\nComparison Summary for Pair", pair_id)
        print("Run Type      | Wall-time (s) | Samples | hearts.get | $ Price | W&B URL")
        print("--------------|--------------|---------|-----------|--------|--------")
        for typ, wandb_data in [("Functional", results["functional_wandb"]), ("Hydra", results["hydra_wandb"])]:
            if wandb_data:
                wall_time = wandb_data["wall_time"] or "NA"
                samples = wandb_data["samples"] or "NA"
                hearts = wandb_data["hearts.get"] or "NA"
                price = wandb_data["price"] or "NA"
                url = wandb_data["url"]
                print(f"{typ:<13} | {wall_time:>12} | {samples:>7} | {hearts:>9} | {price:>6.2f} | {url}")
            else:
                print(f"{typ:<13} | {'NA':>12} | {'NA':>7} | {'NA':>9} | {'NA':>6} | NA")
    except Exception as e:
        logger.error(f"Error in comparison pair {pair_id}: {e}")
        results["error"] = str(e)
    return results


def save_results(results: List[dict], output_file: str):
    """Save comparison results to a file"""
    import json

    # Convert results to JSON-serializable format
    serializable_results = []
    for result in results:
        serializable_result = result.copy()
        # Ensure all values are JSON-serializable
        for key, value in serializable_result.items():
            if not isinstance(value, (str, int, float, bool, type(None))):
                serializable_result[key] = str(value)
        serializable_results.append(serializable_result)

    with open(output_file, "w") as f:
        json.dump(serializable_results, f, indent=2)

    logger.info(f"Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Compare functional vs Hydra training")
    parser.add_argument("--num-pairs", type=int, default=3, help="Number of training pairs to run")
    parser.add_argument("--base-run-name", type=str, default="compare_training", help="Base name for runs")
    parser.add_argument("--output-file", type=str, default="comparison_results.json", help="Output file for results")
    parser.add_argument(
        "--hourly-rate", type=float, default=DEFAULT_HOURLY_RATE, help="Hourly price in USD for cost estimate"
    )
    args = parser.parse_args()
    logger.info(f"Starting training comparison with {args.num_pairs} pairs")
    logger.info(f"Base run name: {args.base_run_name}")
    results = []
    for pair_id in range(1, args.num_pairs + 1):
        logger.info(f"Running comparison pair {pair_id}/{args.num_pairs}")
        pair_results = run_comparison_pair(pair_id, args.base_run_name, args.hourly_rate)
        results.append(pair_results)
        save_results(results, args.output_file)
        logger.info(f"Completed pair {pair_id}")
    # Final summary
    successful_functional = sum(1 for r in results if r.get("functional_exit_code") == 0)
    successful_hydra = sum(1 for r in results if r.get("hydra_exit_code") == 0)
    logger.info("=" * 50)
    logger.info("COMPARISON SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Total pairs: {args.num_pairs}")
    logger.info(f"Successful functional runs: {successful_functional}")
    logger.info(f"Successful Hydra runs: {successful_hydra}")
    logger.info(f"Functional success rate: {successful_functional / args.num_pairs * 100:.1f}%")
    logger.info(f"Hydra success rate: {successful_hydra / args.num_pairs * 100:.1f}%")
    logger.info(f"Results saved to: {args.output_file}")


if __name__ == "__main__":
    main()
