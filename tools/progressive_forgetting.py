#!/usr/bin/env -S uv run
"""Progressive forgetting curriculum experiments.

This script runs progressive curriculum training across all combinations of task sets
to measure catastrophic forgetting, transfer learning, and learning speed.
"""

import argparse
import itertools
import logging
import os
import subprocess
import sys
from typing import Dict, List

import pandas as pd
from omegaconf import OmegaConf

from metta.curriculum.analysis import ForgettingAnalyzer


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def run_single_experiment(
    task_set_1: str,
    task_set_2: str,
    run_id: str,
    base_run_id: str,
    total_timesteps: int = 10_000_000,
    num_workers: int = 4,
    device: str = "auto",
) -> str:
    """Run a single progressive forgetting experiment.

    Args:
        task_set_1: First task set to train on
        task_set_2: Second task set to train on
        run_id: Unique identifier for this run
        base_run_id: Base identifier for runs
        total_timesteps: Total training timesteps
        num_workers: Number of parallel workers
        device: Device to use for training

    Returns:
        Path to the run directory
    """
    logger = logging.getLogger(__name__)

    # Create a temporary config for this specific pair
    temp_dir = "configs/env/mettagrid/curriculum/temp"
    os.makedirs(temp_dir, exist_ok=True)
    config_file_path = f"{temp_dir}/progressive_forgetting_{task_set_1}_{task_set_2}.yaml"
    hydra_config_ref = f"env/mettagrid/curriculum/temp/progressive_forgetting_{task_set_1}_{task_set_2}"

    # Load base config
    logger.info(f"Loading base config from: configs/env/mettagrid/curriculum/progressive_forgetting.yaml")
    base_config = OmegaConf.load("configs/env/mettagrid/curriculum/progressive_forgetting.yaml")
    logger.info(f"Base config loaded successfully. Available task sets: {list(base_config.task_sets.keys())}")

    # Modify task sets to only include the two we want
    if task_set_1 not in base_config.task_sets or task_set_2 not in base_config.task_sets:
        raise ValueError(f"Task sets {task_set_1} or {task_set_2} not found in base config. Available: {list(base_config.task_sets.keys())}")

    # Create modified config with only the two task sets
    modified_config = OmegaConf.create({
        "_target_": base_config._target_,
        "task_sets": {
            task_set_1: base_config.task_sets[task_set_1],
            task_set_2: base_config.task_sets[task_set_2]
        },
        "performance_threshold": base_config.performance_threshold,
        "smoothing": base_config.smoothing,
        "switch_interval": base_config.switch_interval,
        "eval_interval": base_config.eval_interval,
        "randomize_order": base_config.randomize_order,
        "env_overrides": base_config.env_overrides
    })

    # Save modified config
    logger.info(f"Saving modified config to: {config_file_path}")
    with open(config_file_path, "w") as f:
        OmegaConf.save(modified_config, f)

    # Build training command
    cmd = [
        sys.executable,
        "tools/train.py",
        f"run={run_id}",
        f"trainer.curriculum={hydra_config_ref}",
        f"trainer.total_timesteps={total_timesteps}",
        f"trainer.num_workers={num_workers}",
        "trainer.batch_size=2048",  # Fix batch size divisibility issue
        "trainer.minibatch_size=1024",  # Fix minibatch size validation
        "+hardware=macbook",  # Use macbook for local testing
        "device=cpu",  # Override to cpu for local testing
        "wandb.entity=metta-research",  # Specify wandb entity
    ]

    # Only add device override if explicitly specified (not auto)
    if device != "auto":
        cmd.append(f"device={device}")

    logger.info(f"Running experiment: {task_set_1} -> {task_set_2}")
    logger.info(f"Command: {' '.join(cmd)}")

    # Run training
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"Training completed successfully for {run_id}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Training failed for {run_id}: {e}")
        logger.error(f"stdout: {e.stdout}")
        logger.error(f"stderr: {e.stderr}")
        raise

    # Clean up temporary config
    os.remove(config_file_path)

    return f"train_dir/{run_id}"


def analyze_experiment(run_dir: str, task_set_1: str, task_set_2: str) -> Dict[str, float]:
    """Analyze a single experiment to extract forgetting metrics.

    Args:
        run_dir: Path to the training run directory
        task_set_1: First task set name
        task_set_2: Second task set name

    Returns:
        Dictionary of forgetting metrics
    """
    logger = logging.getLogger(__name__)

    analyzer = ForgettingAnalyzer(run_dir)

    try:
        # Load metrics
        metrics_data = analyzer.load_metrics()

        if metrics_data.empty:
            logger.warning(f"No metrics data found in {run_dir}")
            return {}

        # Extract performance trajectories
        performances = analyzer.extract_task_set_performances()

        if not performances:
            logger.warning(f"No performance data found in {run_dir}")
            return {}

        # Calculate forgetting metrics
        forgetting_metrics = analyzer.calculate_forgetting_metrics(performances)

        # Get the specific pair metrics
        pair_name = f"{task_set_1}_to_{task_set_2}"
        if pair_name in forgetting_metrics:
            return forgetting_metrics[pair_name]
        else:
            logger.warning(f"No metrics found for pair {pair_name}")
            return {}

    except Exception as e:
        logger.error(f"Error analyzing experiment {run_dir}: {e}")
        return {}


def run_all_experiments(
    task_sets: List[str], base_run_id: str, total_timesteps: int = 10_000_000, num_workers: int = 4, device: str = "auto"
) -> Dict[str, Dict[str, float]]:
    """Run experiments for all task set pairs.

    Args:
        task_sets: List of task set names
        base_run_id: Base identifier for runs
        total_timesteps: Total training timesteps per experiment
        num_workers: Number of parallel workers
        device: Device to use for training

    Returns:
        Dictionary mapping task set pairs to forgetting metrics
    """
    logger = logging.getLogger(__name__)

    all_metrics = {}

    # Generate all ordered pairs (order matters for forgetting measurement)
    pairs = list(itertools.permutations(task_sets, 2))

    for i, (task_set_1, task_set_2) in enumerate(pairs):
        run_id = f"{base_run_id}_{task_set_1}_to_{task_set_2}"

        logger.info(f"Running experiment {i + 1}/{len(pairs)}: {task_set_1} -> {task_set_2}")

        try:
            # Run training
            run_dir = run_single_experiment(task_set_1, task_set_2, run_id, base_run_id, total_timesteps, num_workers, device)

            # Analyze results
            metrics = analyze_experiment(run_dir, task_set_1, task_set_2)

            pair_name = f"{task_set_1}_to_{task_set_2}"
            all_metrics[pair_name] = metrics

            logger.info(f"Completed experiment {pair_name}: {metrics}")

        except Exception as e:
            logger.error(f"Failed experiment {task_set_1} -> {task_set_2}: {e}")
            all_metrics[f"{task_set_1}_to_{task_set_2}"] = {}

    return all_metrics


def create_summary_report(all_metrics: Dict[str, Dict[str, float]], output_dir: str):
    """Create a summary report of all experiments.

    Args:
        all_metrics: Dictionary of all forgetting metrics
        output_dir: Directory to save the report
    """
    logger = logging.getLogger(__name__)

    os.makedirs(output_dir, exist_ok=True)

    # Convert to DataFrame for easier analysis
    rows = []
    for pair_name, metrics in all_metrics.items():
        row = {"pair": pair_name}
        row.update(metrics)
        rows.append(row)

    df = pd.DataFrame(rows)

    # Save raw data
    df.to_csv(f"{output_dir}/forgetting_metrics.csv", index=False)

    # Create summary statistics
    summary_stats = {}
    for metric in [
        "zero_shot_transfer",
        "forgetting_magnitude",
        "learning_magnitude",
        "learning_speed",
        "forgetting_speed",
    ]:
        if metric in df.columns:
            summary_stats[metric] = {
                "mean": df[metric].mean(),
                "std": df[metric].std(),
                "min": df[metric].min(),
                "max": df[metric].max(),
            }

    # Save summary
    summary_df = pd.DataFrame(summary_stats).T
    summary_df.to_csv(f"{output_dir}/summary_statistics.csv")

    # Create visualizations
    analyzer = ForgettingAnalyzer("")  # Dummy analyzer for plotting

    # Plot forgetting matrix
    analyzer.plot_forgetting_matrix(all_metrics, "forgetting_magnitude", f"{output_dir}/forgetting_matrix.png")

    # Plot learning magnitude matrix
    analyzer.plot_forgetting_matrix(all_metrics, "learning_magnitude", f"{output_dir}/learning_matrix.png")

    # Plot zero-shot transfer matrix
    analyzer.plot_forgetting_matrix(all_metrics, "zero_shot_transfer", f"{output_dir}/zero_shot_transfer_matrix.png")

    logger.info(f"Summary report saved to {output_dir}")

    # Print summary
    print("\n" + "=" * 80)
    print("PROGRESSIVE FORGETTING EXPERIMENT SUMMARY")
    print("=" * 80)

    print(f"\nTotal experiments: {len(all_metrics)} (12 ordered pairs)")
    print(f"Successful experiments: {len([m for m in all_metrics.values() if m])}")

    if summary_stats:
        print("\nSummary Statistics:")
        for metric, stats in summary_stats.items():
            print(f"  {metric}:")
            print(f"    Mean: {stats['mean']:.4f}")
            print(f"    Std:  {stats['std']:.4f}")
            print(f"    Range: [{stats['min']:.4f}, {stats['max']:.4f}]")

    print(f"\nDetailed results saved to: {output_dir}/forgetting_metrics.csv")
    print(f"Visualizations saved to: {output_dir}/")


def main():
    """Main function to run progressive forgetting experiments."""
    setup_logging()
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Run progressive forgetting experiments")
    parser.add_argument(
        "--task-sets",
        nargs="+",
        default=["navigation", "memory", "navigation_sequence", "object_use"],
        help="Task sets to use for experiments",
    )
    parser.add_argument("--base-run-id", default="progressive_forgetting", help="Base identifier for runs")
    parser.add_argument(
        "--total-timesteps", type=int, default=10_000_000, help="Total training timesteps per experiment"
    )
    parser.add_argument("--num-workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--device", default="auto", help="Device to use for training (auto, cpu, cuda)")
    parser.add_argument("--output-dir", default="progressive_forgetting_results", help="Directory to save results")

    args = parser.parse_args()

    logger.info("Starting progressive forgetting experiments")
    logger.info(f"Task sets: {args.task_sets}")
    logger.info(f"Base run ID: {args.base_run_id}")
    logger.info(f"Total timesteps per experiment: {args.total_timesteps}")

    # Run all experiments
    all_metrics = run_all_experiments(
        args.task_sets, args.base_run_id, args.total_timesteps, args.num_workers, args.device
    )

    # Create summary report
    create_summary_report(all_metrics, args.output_dir)

    logger.info("Progressive forgetting experiments completed")


if __name__ == "__main__":
    main()
