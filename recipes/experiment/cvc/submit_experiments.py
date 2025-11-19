#!/usr/bin/env -S uv run
"""Submit variants_curriculum and full_curriculum experiments.

This script submits:
1. Variants curriculum experiments:
   - Single map, all variants
   - 3 small_maps, all variants
   - All maps, all variants
2. Full curriculum experiment:
   - All maps, standard variant
"""

from recipes.experiment.cvc import full_curriculum, variants_curriculum

# Training facility missions (small, focused missions)
SMALL_MAPS = ["harvest", "assemble", "vibe_check"]

# Single map for testing
SINGLE_MAP = ["vibe_check"]

# All maps from full curriculum
ALL_MAPS = list(full_curriculum.FULL_CURRICULUM_MISSIONS)


def submit_variants_experiments():
    """Submit variants_curriculum experiments."""
    print("=" * 80)
    print("Submitting Variants Curriculum Experiments")
    print("=" * 80)

    # 1. Single map, all variants
    print("\n1. Submitting: Single map (vibe_check), all variants")
    variants_curriculum.experiment(
        base_missions=SINGLE_MAP,
        run_name="variants_curriculum_single_map_all_variants",
        skip_git_check=True,
    )

    # 2. 3 small_maps, all variants
    print("\n2. Submitting: 3 small_maps, all variants")
    variants_curriculum.experiment(
        base_missions=SMALL_MAPS,
        run_name="variants_curriculum_3_small_maps_all_variants",
        skip_git_check=True,
    )

    # 3. All maps, all variants
    print("\n3. Submitting: All maps, all variants")
    variants_curriculum.experiment(
        base_missions=ALL_MAPS,
        run_name="variants_curriculum_all_maps_all_variants",
        skip_git_check=True,
    )

    print("\n✓ All variants_curriculum experiments submitted!")


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

