#!/usr/bin/env -S uv run
"""Submit mission-variant curriculum experiments.

This script submits:
1. Variants curriculum experiments (variants="all"):
   - For each mission in DEFAULT_CURRICULUM_MISSIONS: all variants on that single mission
   - All variants on all DEFAULT_CURRICULUM_MISSIONS
2. Full curriculum experiment (variants=None):
   - All maps, standard variant (no variants)
3. S3 successful missions experiment (variants="all"):
   - All variants on missions where S3 policies got reward

1. Tier-based curriculum experiments:
   - Tier 1 (Easy/Proven): go_together, collect_resources_classic, collect_resources_spread
   - Tier 2 (Medium, includes Tier 1): Tier 1 + extractor_hub, divide_and_conquer, harvest, repair, vibe_check
   - Tier 3 (Hard, includes Tier 1+2): Tier 2 + oxygen_bottleneck, energy_starved, collect_far

2. Full curriculum experiments:
   - With diagnostics (baseline)
   - Without diagnostics (test impact of easy missions)
   - Variant-enhanced (add proven variants to Tier 1 missions)

3. Proven variant curricula (for reference):
   - go_together, collect_resources_classic, collect_resources_spread with all variants

Key improvements:
- Progressive deposit rewards: [1.5, 2.0, 2.5, 3.0]
- Adjusted inventory rewards: [0.1, 0.2, 0.3, 0.5]
- Selective variant inclusion (only proven variants)
- Eval environments match training mission:variant combinations

REMOVED experiments (all failed):
- oxygen_bottleneck, energy_starved, collect_far variant-only curricula
- extractor_hub_30/50/70 variant-only curricula
- all_default_missions_all_variants: 300/300 missions failing
"""

from recipes.experiment.cvc import mission_variant_curriculum

# Missions where S3 policies got reward > 0 (from evaluation results)
# Based on evaluation results showing environments where S3 policies succeeded
S3_SUCCESSFUL_EVAL_MISSIONS = [
    "diagnostic_chest_deposit_near",
    "diagnostic_chest_navigation3",
    "diagnostic_extract_missing_oxygen",
    "diagnostic_extract_missing_silicon",
    "easy_hearts",
    "easy_hearts_hello_world",
    "easy_hearts_training",
    "easy_hearts_training_facility",
    "easy_medium_hearts",
    "easy_mode",
    "easy_small_hearts",
    "go_together",
    "repair",
    "single_use_swarm_easy",
    "vibe_check",
]

S3_SUCCESFUL_TRAINING_MISSIONS = [
    "collect_resources_classic",
    "go_together",
    "diagnostic_extract_missing_oxygen",
    "diagnostic_extract_missing_silicon",
    "repair",
    "extractor_hub_30",
    "extractor_hub_50",
    "extractor_hub_70",
]

S3_SUCCESFUL_VARIANTS = [
    "chest_heart_tune",
    "clip_period_on",
    "clipped_oxygen",
    "clipped_silicon",
    "clipping_chaos",
    "cog_tools_only",
    "compass",
    "cyclical_unclip",
    "dark_side",
    "energized",
    "energy_crisis",
    "extractor_heart_tune",
    "heart_chorus",
    "inventory_heart_tune",
    "lonely_heart",
    "pack_rat",
    "resource_bottleneck",
    "rough_terrain",
    "small_50",
    "super_charged",
    "tiny_heart_protocols",
    "vibe_check_min_2",
    "trader"
]

def submit_single_mission_experiments():
    for mission in S3_SUCCESFUL_TRAINING_MISSIONS:
        mission_variant_curriculum.experiment(
        base_missions=[mission],
        run_name=f"single_mission_experiment_{mission}_standard",
        skip_git_check=True,
        variants=None,
        additional_args=[
            "progressive_deposit_rewards=True",
            "adjusted_inventory_rewards=True",
        ],
    )

def submit_all_missions_no_variants():
    mission_variant_curriculum.experiment(
    base_missions=S3_SUCCESFUL_TRAINING_MISSIONS,
    run_name=f"single_mission_experiment_training_missions_standard",
    skip_git_check=True,
    variants=None,
    additional_args=[
        "progressive_deposit_rewards=True",
        "adjusted_inventory_rewards=True",
    ],
    )

    mission_variant_curriculum.experiment(
    base_missions=S3_SUCCESSFUL_EVAL_MISSIONS,
    run_name=f"single_mission_experiment_training_missions_standard",
    skip_git_check=True,
    variants=None,
    additional_args=[
        "progressive_deposit_rewards=True",
        "adjusted_inventory_rewards=True",
    ],
    )

    mission_variant_curriculum.experiment(
    base_missions=S3_SUCCESFUL_TRAINING_MISSIONS +S3_SUCCESSFUL_EVAL_MISSIONS,
    run_name=f"single_mission_experiment_training_missions_standard",
    skip_git_check=True,
    variants=None,
    additional_args=[
        "progressive_deposit_rewards=True",
        "adjusted_inventory_rewards=True",
    ],
    )

def submit_all_missions_with_variants():
    mission_variant_curriculum.experiment(
    base_missions=S3_SUCCESFUL_TRAINING_MISSIONS,
    run_name=f"single_mission_experiment_training_missions_standard",
    skip_git_check=True,
    variants=S3_SUCCESFUL_VARIANTS,
    additional_args=[
        "progressive_deposit_rewards=True",
        "adjusted_inventory_rewards=True",
    ],
    )

    mission_variant_curriculum.experiment(
    base_missions=S3_SUCCESSFUL_EVAL_MISSIONS,
    run_name=f"single_mission_experiment_training_missions_standard",
    skip_git_check=True,
    variants=S3_SUCCESFUL_VARIANTS,
    additional_args=[
        "progressive_deposit_rewards=True",
        "adjusted_inventory_rewards=True",
    ],
    )

    mission_variant_curriculum.experiment(
    base_missions=S3_SUCCESFUL_TRAINING_MISSIONS +S3_SUCCESSFUL_EVAL_MISSIONS,
    run_name=f"single_mission_experiment_training_missions_standard",
    skip_git_check=True,
    variants=S3_SUCCESFUL_VARIANTS,
    additional_args=[
        "progressive_deposit_rewards=True",
        "adjusted_inventory_rewards=True",
    ],
    )

if __name__ == "__main__":
    # submit_variants_experiments()
    # submit_full_curriculum_experiment()
    submit_s3_successful_missions_experiment()
    print("\n" + "=" * 80)
    print("All experiments submitted successfully!")
    print("=" * 80)
