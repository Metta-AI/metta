#!/usr/bin/env -S uv run
"""Submit mission-variant curriculum experiments.

This script submits:
1. Variants curriculum experiments (all_variants_per_mission=True):
   - For each mission in DEFAULT_CURRICULUM_MISSIONS: all variants on that single mission (10 runs)
   - All variants on all DEFAULT_CURRICULUM_MISSIONS (1 run)
2. Full curriculum experiment (all_variants_per_mission=False):
   - All maps, standard variant (no variants)

Note: You can also use mission set names like "eval_missions", "diagnostic_missions",
"training_facility_missions", or "all" instead of individual mission names.
"""

from recipes.experiment import cogs_v_clips
from recipes.experiment.cvc import mission_variant_curriculum

# Default curriculum missions
DEFAULT_MISSIONS = cogs_v_clips.DEFAULT_CURRICULUM_MISSIONS


def submit_variants_experiments():
    """Submit variants curriculum experiments (all_variants_per_mission=True)."""
    print("=" * 80)
    print("Submitting Variants Curriculum Experiments")
    print("=" * 80)

    # 1. For each mission in DEFAULT_CURRICULUM_MISSIONS: all variants on that single mission
    print(f"\nSubmitting {len(DEFAULT_MISSIONS)} single-mission experiments (all variants each):")
    for i, mission in enumerate(DEFAULT_MISSIONS, 1):
        print(f"  {i}. {mission}")
        # Sanitize mission name for run name (replace special chars)
        safe_mission_name = mission.replace("_", "-")
        mission_variant_curriculum.experiment(
            base_missions=[mission],
            run_name=f"variants_curriculum_{safe_mission_name}_all_variants",
            skip_git_check=True,
            all_variants_per_mission=True,
        )

    # 2. All variants on all DEFAULT_CURRICULUM_MISSIONS
    print(f"\nSubmitting: All variants on all {len(DEFAULT_MISSIONS)} DEFAULT_CURRICULUM_MISSIONS")
    mission_variant_curriculum.experiment(
        base_missions=DEFAULT_MISSIONS,
        run_name="variants_curriculum_all_default_missions_all_variants",
        skip_git_check=True,
        all_variants_per_mission=True,
    )

    print(f"\n✓ All {len(DEFAULT_MISSIONS) + 1} variants_curriculum experiments submitted!")


def submit_full_curriculum_experiment():
    """Submit full curriculum experiment with all maps, standard variant (no variants)."""
    print("\n" + "=" * 80)
    print("Submitting Full Curriculum Experiment")
    print("=" * 80)

    print("\nSubmitting: All maps, standard variant (no variants - base missions only)")
    # No variants means standard/base missions only (this is the default)
    mission_variant_curriculum.experiment(
        run_name="full_curriculum_all_maps_standard",
        skip_git_check=True,
        all_variants_per_mission=False,
        variants=None,  # Explicitly no variants
    )

    print("\n✓ Full curriculum experiment submitted!")


if __name__ == "__main__":
    submit_variants_experiments()
    submit_full_curriculum_experiment()
    print("\n" + "=" * 80)
    print("All experiments submitted successfully!")
    print("=" * 80)
