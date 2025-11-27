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

# Proven mission types that succeed with variant-focused training
PROVEN_VARIANT_MISSIONS = [
    "go_together",
    "collect_resources_classic",
    "collect_resources_spread",
]


def submit_tier_experiments():
    """Submit tier-based curriculum experiments."""
    print("=" * 80)
    print("Submitting Tier-Based Curriculum Experiments")
    print("=" * 80)
    print("\nTier structure (cumulative):")
    print("  Tier 1: Easy/Proven missions (go_together, collect_resources variants)")
    print("  Tier 2: Tier 1 + Medium missions (extractor_hub, divide_and_conquer, etc.)")
    print("  Tier 3: Tier 2 + Hard missions (oxygen_bottleneck, energy_starved, collect_far)")

    tiers = [
        ("tier1", "Tier 1 (Easy/Proven)"),
        ("tier2", "Tier 2 (Medium, includes Tier 1)"),
        ("tier3", "Tier 3 (Hard, includes Tier 1+2)"),
    ]

    for tier_name, tier_desc in tiers:
        print(f"\nSubmitting: {tier_desc}")
        mission_variant_curriculum.experiment(
            base_missions=[tier_name],
            run_name=f"curriculum_{tier_name}_v2",
            skip_git_check=True,
            all_variants_per_mission=False,
            variants=None,
            additional_args=[
                "progressive_deposit_rewards=True",
                "adjusted_inventory_rewards=True",
            ],
        )

    print("\n✓ All tier-based experiments submitted!")


def submit_variants_experiments():
    """Submit variants curriculum experiments (variants="all")."""
    print("=" * 80)
    print("Submitting Variants Curriculum Experiments")
    print("=" * 80)
    print("\nThese achieved 100% success in previous runs - submitting for reference:")
    print("  - go_together: 30/30 variants mastered")
    print("  - collect_resources_classic: 30/30 variants mastered")
    print("  - collect_resources_spread: 30/30 variants mastered")

    # Only submit for proven mission types
    print(f"\nSubmitting {len(PROVEN_VARIANT_MISSIONS)} proven single-mission experiments:")
    for i, mission in enumerate(PROVEN_VARIANT_MISSIONS, 1):
        print(f"  {i}. {mission}")
        safe_mission_name = mission.replace("_", "-")
        mission_variant_curriculum.experiment(
            base_missions=[mission],
            run_name=f"variants_curriculum_{safe_mission_name}_all_variants_v3",
            skip_git_check=True,
            variants="all",
        )

    # 2. All variants on all DEFAULT_CURRICULUM_MISSIONS
    print(f"\nSubmitting: All variants on all {len(DEFAULT_MISSIONS)} DEFAULT_CURRICULUM_MISSIONS")
    mission_variant_curriculum.experiment(
        base_missions=DEFAULT_MISSIONS,
        run_name="variants_curriculum_all_default_missions_all_variants",
        skip_git_check=True,
        variants="all",
    )

    print(f"\n✓ All {len(DEFAULT_MISSIONS) + 1} variants_curriculum experiments submitted!")


def submit_full_curriculum_experiments():
    """Submit full curriculum experiments with different configurations.

    Tests impact of:
    - Including/excluding diagnostic missions
    - Adding proven variants to Tier 1 missions
    """
    print("\n" + "=" * 80)
    print("Submitting Full Curriculum Experiments (PRIMARY ITERATION TARGET)")
    print("=" * 80)

    # Experiment 1: Full curriculum WITH diagnostics (baseline)
    print("\n1. Full curriculum WITH diagnostics (baseline):")
    print("   - All missions including diagnostics")
    print("   - Progressive deposit rewards: [1.5, 2.0, 2.5, 3.0]")
    print("   - Adjusted inventory rewards: [0.1, 0.2, 0.3, 0.5]")
    mission_variant_curriculum.experiment(
        run_name="full_curriculum_all_maps_with_diagnostics_v2",
        skip_git_check=True,
        variants=None,  # Explicitly no variants
    )

    # Experiment 2: Full curriculum WITHOUT diagnostics
    print("\n2. Full curriculum WITHOUT diagnostics:")
    print("   - All missions except diagnostics (may improve diversity)")
    print("   - Same reward configuration")
    mission_variant_curriculum.experiment(
        base_missions=["all_no_diagnostics"],
        run_name="full_curriculum_all_maps_no_diagnostics_v2",
        skip_git_check=True,
        all_variants_per_mission=False,
        variants=None,
        additional_args=[
            "progressive_deposit_rewards=True",
            "adjusted_inventory_rewards=True",
        ],
    )

    # Experiment 3: Variant-enhanced full curriculum
    # Note: This would require custom logic to add variants only to Tier 1 missions
    # For now, we'll add variants to all missions but only proven ones
    print("\n3. Variant-enhanced full curriculum:")
    print("   - All missions (with diagnostics)")
    print("   - Add proven variants to all missions (selective variant inclusion)")
    print("   - Tests selective variant inclusion")
    # Use all_variants_per_mission=False with proven variants only
    # This applies proven variants to all missions (simpler than tier-specific)
    mission_variant_curriculum.experiment(
        run_name="full_curriculum_variant_enhanced_v2",
        skip_git_check=True,
        all_variants_per_mission=False,
        variants=list(mission_variant_curriculum.PROVEN_VARIANTS[:10]),  # Use subset for testing
        additional_args=[
            "progressive_deposit_rewards=True",
            "adjusted_inventory_rewards=True",
        ],
    )

    print("\n✓ All full curriculum experiments submitted!")


def submit_full_curriculum_with_dense():
    """Submit full curriculum experiments that include dense environments.

    This combines the full curriculum missions with dense training environments
    to provide a comprehensive training curriculum.
    """
    print("\n" + "=" * 80)
    print("Submitting Full Curriculum + Dense Environments Experiments")
    print("=" * 80)
    print("\nThis combines:")
    print("  - Full curriculum missions (all maps, variants, diagnostics)")
    print("  - Dense training environments (resource reduction + max_uses bucketing)")

    import base64
    import json

    from recipes.experiment.cvc.dense_curriculum import DENSE_TRAINING_MISSIONS

    # Default dense curriculum parameters
    dense_resource_levels = [7, 8, 9, 10]  # Dense levels
    dense_maps = [m.name for m in DENSE_TRAINING_MISSIONS]  # All maps
    dense_max_uses = [1, 2, 3, 5, 10, 255]

    # Encode parameters as base64 for command-line passing
    dense_resource_levels_b64 = base64.b64encode(json.dumps(dense_resource_levels).encode()).decode()
    dense_maps_b64 = base64.b64encode(json.dumps(dense_maps).encode()).decode()
    dense_max_uses_b64 = base64.b64encode(json.dumps(dense_max_uses).encode()).decode()

    print("\n1. Full curriculum + Dense environments (with diagnostics):")
    print("   - All full curriculum missions")
    print("   - Dense environments: levels [7,8,9,10], all maps, max_uses bucketing")
    print("   - Includes diagnostic missions in both curricula")
    mission_variant_curriculum.experiment(
        run_name="full_curriculum_with_dense_v1",
        skip_git_check=True,
        all_variants_per_mission=False,
        variants=None,
        additional_args=[
            "progressive_deposit_rewards=True",
            "adjusted_inventory_rewards=True",
            f"dense_resource_levels_b64={dense_resource_levels_b64}",
            f"dense_maps_to_use_b64={dense_maps_b64}",
            "dense_include_diagnostics=True",
            f"dense_max_uses_values_b64={dense_max_uses_b64}",
            "use_combined_curriculum=True",
        ],
    )

    print("\n✓ Full curriculum + Dense environments experiment submitted!")


def submit_s3_successful_missions_experiment():
    """Submit curriculum experiment with only missions where S3 policies got reward, all variants."""
    print("\n" + "=" * 80)
    print("Submitting S3 Successful Missions Curriculum Experiment")
    print("=" * 80)

    # Missions where S3 policies got reward > 0 (from evaluation results)
    # Based on evaluation results showing environments where S3 policies succeeded
    S3_SUCCESSFUL_MISSIONS = [
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

    print(f"\nSubmitting: All variants on {len(S3_SUCCESSFUL_MISSIONS)} missions where S3 policies got reward")
    print("Missions:")
    for i, mission in enumerate(S3_SUCCESSFUL_MISSIONS, 1):
        print(f"  {i}. {mission}")

    mission_variant_curriculum.experiment(
        base_missions=S3_SUCCESSFUL_MISSIONS,
        run_name="variants_curriculum_11-26.3",
        skip_git_check=True,
        variants="all",  # Curriculum over all variants
    )

    print("\n✓ S3 successful missions curriculum experiment submitted!")


if __name__ == "__main__":
    # submit_variants_experiments()
    # submit_full_curriculum_experiment()
    submit_s3_successful_missions_experiment()
    print("\n" + "=" * 80)
    print("All experiments submitted successfully!")
    print("=" * 80)
