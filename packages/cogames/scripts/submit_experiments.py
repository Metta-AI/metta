#!/usr/bin/env -S uv run
"""Submit mission-variant curriculum experiments.

Based on analysis (see run_analysis/README.md and NEXT_BATCH_TRAINING_STRATEGY.md), this script submits:

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
    """Submit variants curriculum experiments only for proven mission types (for reference)."""
    print("\n" + "=" * 80)
    print("Submitting Proven Variant Curriculum Experiments (Reference)")
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
            run_name=f"variants_curriculum_{safe_mission_name}_all_variants_v2",
            skip_git_check=True,
            all_variants_per_mission=True,
            additional_args=[
                "progressive_deposit_rewards=True",
                "adjusted_inventory_rewards=True",
            ],
        )

    print(f"\n✓ All {len(PROVEN_VARIANT_MISSIONS)} proven variants_curriculum experiments submitted!")


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
        all_variants_per_mission=False,
        variants=None,
        additional_args=[
            "progressive_deposit_rewards=True",
            "adjusted_inventory_rewards=True",
        ],
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


if __name__ == "__main__":
    submit_tier_experiments()
    submit_full_curriculum_experiments()
    submit_variants_experiments()  # Reference runs
    print("\n" + "=" * 80)
    print("All experiments submitted successfully!")
    print("=" * 80)
