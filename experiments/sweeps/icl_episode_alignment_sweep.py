#!/usr/bin/env python3
"""
Advanced BPTT horizon sweep with episode boundary alignment strategies.
This version ensures proper alignment with episode boundaries to avoid
truncating episodes at arbitrary points.
"""

import subprocess
import time
from dataclasses import dataclass
from typing import List, Optional

# Memory budget constant
MEMORY_BUDGET = 2064384 * 256  # ~528M tokens


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""

    horizon: int
    batch_size: int
    name: str
    episode_alignment: Optional[str] = None  # 'pack', 'pad', or None
    pack_episodes: Optional[int] = None  # Number of episodes to pack together
    description: str = ""
    minibatch_size: Optional[int] = None


def get_experiment_configs() -> List[ExperimentConfig]:
    """Define all experiment configurations with different alignment strategies."""
    configs = []

    # Baseline configurations (no special alignment)
    configs.extend(
        [
            ExperimentConfig(
                horizon=128,
                batch_size=4128768,
                name="h128_baseline",
                description="Half episode view, may miss late-episode learning",
            ),
            ExperimentConfig(
                horizon=256,
                batch_size=2064384,
                name="h256_baseline",
                description="Matches short episodes, truncates long ones",
            ),
            ExperimentConfig(
                horizon=512,
                batch_size=1032192,
                name="h512_baseline",
                description="Captures all episodes fully",
            ),
        ]
    )

    # Episode-aligned configurations
    configs.extend(
        [
            # Pack exactly 2 short episodes (256 steps each) into 512 horizon
            ExperimentConfig(
                horizon=512,
                batch_size=1032192,
                name="h512_pack2",
                episode_alignment="pack",
                pack_episodes=2,
                description="Pack exactly 2 short episodes for cross-episode learning",
            ),
            # Use 768 to fit 3 short episodes or 1.5 long episodes
            ExperimentConfig(
                horizon=768,
                batch_size=688128,
                name="h768_pack3",
                episode_alignment="pack",
                pack_episodes=3,
                description="Pack 3 short episodes, enables pattern recognition across episodes",
                minibatch_size=49152,  # 768 * 64; divides batch cleanly and satisfies divisibility
            ),
            # Use 1024 to fit exactly 2 long episodes (512 steps each)
            ExperimentConfig(
                horizon=1024,
                batch_size=524288,  # Adjusted to satisfy divisibility with default minibatch
                name="h1024_pack2long",
                episode_alignment="pack",
                pack_episodes=2,
                description="Pack exactly 2 long episodes",
            ),
            # Pad short episodes to 512 boundary
            ExperimentConfig(
                horizon=512,
                batch_size=1032192,
                name="h512_pad",
                episode_alignment="pad",
                description="Pad short episodes to 512, ensuring full episode visibility",
            ),
        ]
    )

    return configs


def launch_job(
    config: ExperimentConfig,
    replicate: int,
    group: str,
    dry_run: bool = False,
) -> subprocess.CompletedProcess:
    """Launch a single Skypilot job with the given configuration."""

    job_name = f"icl_align_{config.name}_rep{replicate}"

    # Build the command
    cmd = [
        "sky",
        "launch",
        "--name",
        job_name,
        "--gpus",
        "A100:1",
        "--use-spot",
        "--down",
        "--detach-setup",
        "--detach-run",
        "--",
        "uv",
        "run",
        "./tools/run.py",
        "experiments.recipes.icl_resource_chain.train",
        f"run={job_name}",
        f"group={group}",
        f"trainer.bptt_horizon={config.horizon}",
        f"trainer.batch_size={config.batch_size}",
    ]

    # Add minibatch override if specified
    if config.minibatch_size:
        cmd.append(f"trainer.minibatch_size={config.minibatch_size}")

    # Add alignment-specific parameters
    if config.episode_alignment:
        cmd.append(f"trainer.episode_alignment={config.episode_alignment}")
    if config.pack_episodes:
        cmd.append(f"trainer.pack_episodes={config.pack_episodes}")

    # Add wandb logging
    cmd.extend(
        [
            f"wandb.config.bptt_horizon={config.horizon}",
            f"wandb.config.batch_size={config.batch_size}",
            f"wandb.config.minibatch_size={config.minibatch_size or 16384}",
            f"wandb.config.memory_budget={config.batch_size * config.horizon}",
            f"wandb.config.episode_alignment={config.episode_alignment or 'none'}",
            f"wandb.config.pack_episodes={config.pack_episodes or 0}",
            f"wandb.config.replicate={replicate}",
            f"wandb.config.experiment_name={config.name}",
        ]
    )

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Launching {job_name}...")
    print(f"  Config: {config.name}")
    print(f"  Description: {config.description}")
    print(f"  Horizon: {config.horizon}, Batch size: {config.batch_size:,}")
    if config.minibatch_size:
        print(f"  Minibatch size: {config.minibatch_size:,}")
    print(f"  Memory budget: {config.batch_size * config.horizon:,} tokens")
    if config.episode_alignment:
        print(f"  Alignment: {config.episode_alignment}")
        if config.pack_episodes:
            print(f"  Pack episodes: {config.pack_episodes}")

    if dry_run:
        print(f"  Command: {' '.join(cmd)}")
        return subprocess.CompletedProcess(args=cmd, returncode=0)

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"ERROR launching {job_name}:")
        print(result.stderr)
    else:
        print(f"Successfully launched {job_name}")

    return result


def main():
    """Launch all jobs in the sweep."""
    import argparse

    parser = argparse.ArgumentParser(description="Launch episode-aligned BPTT sweep")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show commands without launching"
    )
    parser.add_argument(
        "--replicates", type=int, default=3, help="Number of replicates per config"
    )
    parser.add_argument(
        "--filter", type=str, help="Only run configs matching this substring"
    )
    args = parser.parse_args()

    configs = get_experiment_configs()

    # Filter configs if requested
    if args.filter:
        configs = [c for c in configs if args.filter in c.name]

    group_name = f"icl_align_sweep_{int(time.time())}"

    print("Episode-Aligned BPTT Horizon Sweep")
    print(f"Group: {group_name}")
    print(f"Configurations: {len(configs)}")
    print(f"Replicates per config: {args.replicates}")
    print(f"Total jobs: {len(configs) * args.replicates}")
    print(f"Dry run: {args.dry_run}")

    print("\nConfigurations:")
    for config in configs:
        print(f"\n  {config.name}:")
        print(f"    Horizon: {config.horizon}, Batch: {config.batch_size:,}")
        print(f"    {config.description}")

    if not args.dry_run:
        response = input("\nProceed with launching? (y/N): ")
        if response.lower() != "y":
            print("Aborted.")
            return

    successful_jobs = []
    failed_jobs = []

    # Launch all jobs
    for config in configs:
        for rep in range(1, args.replicates + 1):
            result = launch_job(config, rep, group_name, dry_run=args.dry_run)

            job_info = f"{config.name}_rep{rep}"
            if result.returncode == 0:
                successful_jobs.append(job_info)
            else:
                failed_jobs.append((job_info, result.stderr))

            if not args.dry_run:
                time.sleep(2)  # Delay between launches

    # Summary
    print("\n" + "=" * 60)
    print("LAUNCH SUMMARY")
    print("=" * 60)
    print(
        f"Total jobs {'would be' if args.dry_run else ''} attempted: {len(configs) * args.replicates}"
    )

    if not args.dry_run:
        print(f"Successful launches: {len(successful_jobs)}")
        print(f"Failed launches: {len(failed_jobs)}")

        if failed_jobs:
            print("\nFailed jobs:")
            for job, error in failed_jobs:
                print(f"  - {job}")
                if error:
                    print(f"    Error: {error.splitlines()[0]}")

        print(f"\nWandB group: {group_name}")
        print("\nMonitoring commands:")
        print("  sky status                    # View all jobs")
        print("  sky logs <job_name>          # View specific job logs")
        print("  sky cancel 'icl_align_*'     # Cancel all alignment jobs")
        print("\nWandB dashboard:")
        print(f"  Filter by group: {group_name}")


if __name__ == "__main__":
    main()
