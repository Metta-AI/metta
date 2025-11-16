#!/usr/bin/env python3
"""Test script to verify dual-pool configuration is set up correctly."""

import sys

sys.path.insert(0, "/Users/bullm/Documents/GitHub/metta")

from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressConfig
from metta.cogworks.curriculum.task_tracker import DualPoolTaskTracker

# Create the same config as variant_maps.py
algorithm_config = LearningProgressConfig.default_dual_pool(
    num_explore_tasks=56,
    num_exploit_tasks=200,
    min_samples_for_lp=5,
    promotion_min_samples=5,
)

print("=" * 80)
print("DUAL-POOL CONFIGURATION CHECK")
print("=" * 80)
print()

# Check basic configuration
print(f"✓ use_dual_pool: {algorithm_config.use_dual_pool}")
print(f"✓ num_explore_tasks: {algorithm_config.num_explore_tasks}")
print(f"✓ num_exploit_tasks: {algorithm_config.num_exploit_tasks}")
print(f"✓ num_active_tasks: {algorithm_config.num_active_tasks}")
print(f"✓ promotion_min_samples: {algorithm_config.promotion_min_samples}")
print(f"✓ min_samples_for_lp: {algorithm_config.min_samples_for_lp}")
print()

# Check EER configuration
print("EER Configuration:")
print(f"  explore_exploit_ratio_init: {algorithm_config.explore_exploit_ratio_init}")
print(f"  explore_exploit_ratio_min: {algorithm_config.explore_exploit_ratio_min}")
print(f"  explore_exploit_ratio_max: {algorithm_config.explore_exploit_ratio_max}")
print(f"  explore_exploit_ratio_alpha: {algorithm_config.explore_exploit_ratio_alpha}")
print(f"  promotion_rate_window: {algorithm_config.promotion_rate_window}")
print()

# Create algorithm and check TaskTracker type
print("Creating algorithm...")
algorithm = algorithm_config.create(num_tasks=256)
print(f"✓ Algorithm created: {type(algorithm).__name__}")
print(f"✓ TaskTracker type: {type(algorithm.task_tracker).__name__}")
print(f"✓ Is DualPoolTaskTracker: {isinstance(algorithm.task_tracker, DualPoolTaskTracker)}")
print()

# Test stats collection
print("Testing stats collection...")
stats = algorithm.get_detailed_stats()
dual_pool_stats = {k: v for k, v in stats.items() if "dual_pool" in k}

if dual_pool_stats:
    print(f"✅ SUCCESS: Found {len(dual_pool_stats)} dual-pool metrics:")
    for key, value in sorted(dual_pool_stats.items()):
        print(f"  - {key}: {value}")
else:
    print("❌ ERROR: No dual-pool metrics found!")
    print(f"Available stats keys: {list(stats.keys())[:10]}...")
print()

# Test with algorithm/ prefix (as it would appear in wandb)
algorithm_stats = algorithm.stats("algorithm/")
dual_pool_algorithm_stats = {k: v for k, v in algorithm_stats.items() if "dual_pool" in k}

print("With algorithm/ prefix (as in wandb):")
if dual_pool_algorithm_stats:
    print(f"✅ SUCCESS: Found {len(dual_pool_algorithm_stats)} metrics:")
    for key in sorted(dual_pool_algorithm_stats.keys()):
        print(f"  - {key}")
else:
    print("❌ ERROR: No dual-pool metrics found with prefix!")
print()

print("=" * 80)
print("VERIFICATION COMPLETE")
print("=" * 80)
