#!/usr/bin/env python3
"""
Script to run navigation evaluations on specific policy URIs
"""

import os
import subprocess
import sys


def run_navigation_eval(policy_uri: str, device: str = "cpu") -> None:
    """Run navigation evaluation for a specific policy URI"""
    cmd = [
        "./tools/sim.py",
        "sim=navigation",
        "run=navigation101",
        f"policy_uri={policy_uri}",
        "sim_job.stats_db_uri=wandb://stats/navigation_db",
        f"device={device}",
    ]

    print(f"Running evaluation for: {policy_uri}")
    print(f"Command: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True, cwd=os.getcwd())
        print(f"✓ Completed evaluation for: {policy_uri}")
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed evaluation for: {policy_uri}")
        print(f"Error: {e}")
        return False

    return True


def main():
    # List of policy URIs to evaluate
    policy_uris = [
        "wandb://run/jacke.sky_comprehensive_20250716_110454",
        "wandb://run/jacke.sky_comprehensive_20250716_112035",
        "wandb://run/jacke.sky_comprehensive_20250716_134110",
        "wandb://run/jacke.sky_comprehensive_20250716_145441",
        "wandb://run/jacke.nav_extended_multiroom_20250714_150559",
        "wandb://run/jacke.nav_extended_multiroom_20250715_112749",
        "wandb://run/jacke.nav_extended_multiroom_20250715_112755",
        "wandb://run/jacke.sky_raster_standard_20250716_110514",
        "wandb://run/jacke.sky_raster_standard_20250716_112030",
        "wandb://run/jacke.sky_raster_standard_20250716_134153",
        "wandb://run/jacke.sky_raster_standard_20250716_145423",
        "wandb://run/jacke.sky_spiral_traditional_20250716_110438",
        "wandb://run/jacke.sky_spiral_traditional_20250716_112045",
        "wandb://run/jacke.sky_spiral_traditional_20250716_134205",
        "wandb://run/jacke.sky_spiral_traditional_20250716_145416",
        "wandb://run/jacke.sky_raster_nav_20250716_171723",
        "wandb://run/jacke.sky_spiral_raster_only_20250716_112801",
        "wandb://run/jacke.sky_spiral_raster_only_20250716_112805",
        "wandb://run/jacke.sky_spiral_raster_only_20250716_134139",
        "wandb://run/jacke.sky_spiral_raster_only_20250716_145434",
        "wandb://run/jacke.sky_spiral_nav_20250716_171713",
    ]

    device = "cpu"
    if len(sys.argv) > 1:
        device = sys.argv[1]

    print(f"Running navigation evaluations on {len(policy_uris)} runs using device: {device}")
    print("=" * 60)

    successful = 0
    failed = 0

    for i, policy_uri in enumerate(policy_uris, 1):
        print(f"\n[{i}/{len(policy_uris)}] Processing: {policy_uri}")
        if run_navigation_eval(policy_uri, device):
            successful += 1
        else:
            failed += 1

    print("\n" + "=" * 60)
    print("Evaluation Summary:")
    print(f"✓ Successful: {successful}")
    print(f"✗ Failed: {failed}")
    print(f"Total: {len(policy_uris)}")


if __name__ == "__main__":
    main()
