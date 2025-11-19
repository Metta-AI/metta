#!/usr/bin/env -S uv run
"""Submit variants_curriculum and full_curriculum experiments.

This script submits:
1. Variants curriculum experiments:
   - For each mission in DEFAULT_CURRICULUM_MISSIONS: all variants on that single mission (10 runs)
   - All variants on all DEFAULT_CURRICULUM_MISSIONS (1 run)
2. Full curriculum experiment:
   - All maps, standard variant
"""

from recipes.experiment import cogs_v_clips
from recipes.experiment.cvc import full_curriculum, variants_curriculum

# Default curriculum missions
DEFAULT_MISSIONS = cogs_v_clips.DEFAULT_CURRICULUM_MISSIONS


def submit_variants_experiments():
    """Submit variants_curriculum experiments."""
    print("=" * 80)
    print("Submitting Variants Curriculum Experiments")
    print("=" * 80)

    # 1. For each mission in DEFAULT_CURRICULUM_MISSIONS: all variants on that single mission
    print(f"\nSubmitting {len(DEFAULT_MISSIONS)} single-mission experiments (all variants each):")
    for i, mission in enumerate(DEFAULT_MISSIONS, 1):
        print(f"  {i}. {mission}")
        # Sanitize mission name for run name (replace special chars)
        safe_mission_name = mission.replace("_", "-")
        variants_curriculum.experiment(
            base_missions=[mission],
            run_name=f"variants_curriculum_{safe_mission_name}_all_variants",
            skip_git_check=True,
        )

    # 2. All variants on all DEFAULT_CURRICULUM_MISSIONS
    print(f"\nSubmitting: All variants on all {len(DEFAULT_MISSIONS)} DEFAULT_CURRICULUM_MISSIONS")
    variants_curriculum.experiment(
        base_missions=DEFAULT_MISSIONS,
        run_name="variants_curriculum_all_default_missions_all_variants",
        skip_git_check=True,
    )

    print(f"\n✓ All {len(DEFAULT_MISSIONS) + 1} variants_curriculum experiments submitted!")


def submit_full_curriculum_experiment():
    """Submit full_curriculum experiment with all maps, standard variant (no variants)."""
    print("\n" + "=" * 80)
    print("Submitting Full Curriculum Experiment")
    print("=" * 80)

    print("\nSubmitting: All maps, standard variant (no variants - base missions only)")
    # No variants means standard/base missions only (this is the default)
    full_curriculum.experiment(
        run_name="full_curriculum_all_maps_standard",
        skip_git_check=True,
    )

    print("\n✓ Full curriculum experiment submitted!")


if __name__ == "__main__":
    submit_variants_experiments()
    submit_full_curriculum_experiment()
    print("\n" + "=" * 80)
    print("All experiments submitted successfully!")
    print("=" * 80)

