import time

from cogames.cogs_vs_clips.navigation_missions import NAVIGATION_MISSIONS
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
    "easy_hearts_training_facility",
    "repair",
    "vibe_check",
]

S3_SUCCESFUL_TRAINING_MISSIONS = [
    "diagnostic_extract_missing_oxygen",
    "diagnostic_extract_missing_silicon",
    "repair",
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
    "trader",
]


def submit_single_mission_experiments():
    """Submit individual mission experiments (no variants)."""
    for mission in S3_SUCCESFUL_TRAINING_MISSIONS:
        mission_variant_curriculum.experiment(
            base_missions=[mission],
            run_name=f"{mission}_base",
            skip_git_check=True,
            variants=None,
        )


def submit_all_missions_no_variants():
    """Submit S3 successful missions (no variants)."""
    date = time.strftime(".%m%d")

    mission_variant_curriculum.experiment(
        base_missions=S3_SUCCESFUL_TRAINING_MISSIONS,
        run_name=f"s3train_base{date}",
        skip_git_check=True,
        variants=None,
    )

    mission_variant_curriculum.experiment(
        base_missions=S3_SUCCESSFUL_EVAL_MISSIONS,
        run_name=f"s3eval_base{date}",
        skip_git_check=True,
        variants=None,
    )

    mission_variant_curriculum.experiment(
        base_missions=S3_SUCCESFUL_TRAINING_MISSIONS + S3_SUCCESSFUL_EVAL_MISSIONS,
        run_name=f"s3all_base{date}",
        skip_git_check=True,
        variants=None,
    )


def submit_all_missions_with_variants():
    """Submit S3 successful missions with S3 successful variants."""
    date = time.strftime(".%m%d")

    mission_variant_curriculum.experiment(
        base_missions=S3_SUCCESFUL_TRAINING_MISSIONS,
        run_name=f"s3train_s3vars{date}",
        skip_git_check=True,
        variants=S3_SUCCESFUL_VARIANTS,
    )

    mission_variant_curriculum.experiment(
        base_missions=S3_SUCCESSFUL_EVAL_MISSIONS,
        run_name=f"s3eval_s3vars{date}",
        skip_git_check=True,
        variants=S3_SUCCESFUL_VARIANTS,
    )

    mission_variant_curriculum.experiment(
        base_missions=S3_SUCCESFUL_TRAINING_MISSIONS + S3_SUCCESSFUL_EVAL_MISSIONS,
        run_name=f"s3all_s3vars{date}",
        skip_git_check=True,
        variants=S3_SUCCESFUL_VARIANTS,
    )


def submit_full_curriculum_experiment_standard():
    """Submit full curriculum (no variants)."""
    date = time.strftime(".%m%d")
    mission_variant_curriculum.experiment(
        base_missions=None,
        run_name=f"full_base{date}",
        skip_git_check=True,
        variants=None,
    )


def submit_full_curriculum_experiment_all_variants():
    """Submit full curriculum with all variants."""
    date = time.strftime(".%m%d")
    mission_variant_curriculum.experiment(
        base_missions=None,
        run_name=f"full_all{date}",
        skip_git_check=True,
        variants="all",
    )


def submit_full_curriculum_with_navigation():
    """Submit full curriculum + navigation missions (no variants)."""
    from recipes.experiment.cvc.mission_variant_curriculum import FULL_CURRICULUM_MISSIONS

    date = time.strftime(".%m%d")
    nav_mission_names = [m.name for m in NAVIGATION_MISSIONS]
    combined_missions = list(FULL_CURRICULUM_MISSIONS) + nav_mission_names

    mission_variant_curriculum.experiment(
        base_missions=combined_missions,
        run_name=f"fullnav_base{date}",
        skip_git_check=True,
        variants=None,
    )


def submit_full_curriculum_with_navigation_with_variants():
    """Submit full curriculum + navigation missions with all variants."""
    from recipes.experiment.cvc.mission_variant_curriculum import FULL_CURRICULUM_MISSIONS

    date = time.strftime(".%m%d")
    nav_mission_names = [m.name for m in NAVIGATION_MISSIONS]
    combined_missions = list(FULL_CURRICULUM_MISSIONS) + nav_mission_names

    mission_variant_curriculum.experiment(
        base_missions=combined_missions,
        run_name=f"fullnav_all{date}",
        skip_git_check=True,
        variants="all",
    )


def submit_missions_with_navigation():
    """Submit S3 subset + navigation missions (no variants)."""
    date = time.strftime(".%m%d")
    nav_mission_names = [m.name for m in NAVIGATION_MISSIONS]
    combined_missions = S3_SUCCESFUL_TRAINING_MISSIONS + S3_SUCCESSFUL_EVAL_MISSIONS + nav_mission_names

    mission_variant_curriculum.experiment(
        base_missions=combined_missions,
        run_name=f"subsetnav_base{date}",
        skip_git_check=True,
        variants=None,
    )


def submit_missions_with_navigation_with_variants():
    """Submit S3 subset + navigation missions with all variants."""
    date = time.strftime(".%m%d")
    nav_mission_names = [m.name for m in NAVIGATION_MISSIONS]
    combined_missions = S3_SUCCESFUL_TRAINING_MISSIONS + S3_SUCCESSFUL_EVAL_MISSIONS + nav_mission_names

    mission_variant_curriculum.experiment(
        base_missions=combined_missions,
        run_name=f"subsetnav_all{date}",
        skip_git_check=True,
        variants="all",
    )


def submit_proc_gen_experiments():
    """Submit procedural generation missions experiments."""
    date = time.strftime(".%m%d")

    # Proc gen missions only (no variants)
    mission_variant_curriculum.experiment(
        base_missions=["proc_gen_missions"],
        run_name=f"procgen_base{date}",
        skip_git_check=True,
        variants=None,
    )

    # Proc gen missions with all variants
    mission_variant_curriculum.experiment(
        base_missions=["proc_gen_missions"],
        run_name=f"procgen_all{date}",
        skip_git_check=True,
        variants="all",
    )

    # Proc gen + diagnostics (no variants)
    mission_variant_curriculum.experiment(
        base_missions=["proc_gen_missions", "diagnostic_missions"],
        run_name=f"procgen_diag_base{date}",
        skip_git_check=True,
        variants=None,
    )

    # Proc gen + diagnostics with all variants
    mission_variant_curriculum.experiment(
        base_missions=["proc_gen_missions", "diagnostic_missions"],
        run_name=f"procgen_diag_all{date}",
        skip_git_check=True,
        variants="all",
    )


if __name__ == "__main__":
    # submit_single_mission_experiments()
    # submit_all_missions_no_variants()
    # submit_all_missions_with_variants()
    # submit_full_curriculum_experiment_standard()
    # submit_full_curriculum_experiment_all_variants()

    # submit_full_curriculum_with_navigation()
    # submit_full_curriculum_with_navigation_with_variants()
    submit_missions_with_navigation()
    # submit_missions_with_navigation_with_variants()

    # submit_proc_gen_experiments()

    print("\n" + "=" * 80)
    print("All experiments submitted successfully!")
    print("=" * 80)
