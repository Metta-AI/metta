from recipes.experiment.cvc import mission_variant_curriculum
from cogames.cogs_vs_clips.navigation_missions import NAVIGATION_MISSIONS

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
    "repair",
    "single_use_swarm_easy",
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
    "trader"
]

def submit_single_mission_experiments():
    for mission in S3_SUCCESFUL_TRAINING_MISSIONS:
        mission_variant_curriculum.experiment(
        base_missions=[mission],
        run_name=f"single_mission_experiment_{mission}_standard",
        skip_git_check=True,
        variants=None,
    )

def submit_all_missions_no_variants():
    mission_variant_curriculum.experiment(
    base_missions=S3_SUCCESFUL_TRAINING_MISSIONS,
    run_name=f"single_mission_experiment_training_missions_standard",
    skip_git_check=True,
    variants=None,
    )

    mission_variant_curriculum.experiment(
    base_missions=S3_SUCCESSFUL_EVAL_MISSIONS,
    run_name=f"single_mission_experiment_training_missions_standard",
    skip_git_check=True,
    variants=None,
    )

    mission_variant_curriculum.experiment(
    base_missions=S3_SUCCESFUL_TRAINING_MISSIONS +S3_SUCCESSFUL_EVAL_MISSIONS,
    run_name=f"single_mission_experiment_training_missions_standard",
    skip_git_check=True,
    variants=None,
    )

def submit_all_missions_with_variants():
    mission_variant_curriculum.experiment(
    base_missions=S3_SUCCESFUL_TRAINING_MISSIONS,
    run_name=f"single_mission_experiment_training_missions_all_variants",
    skip_git_check=True,
    variants=S3_SUCCESFUL_VARIANTS,
    )

    mission_variant_curriculum.experiment(
    base_missions=S3_SUCCESSFUL_EVAL_MISSIONS,
    run_name=f"single_mission_experiment_training_missions_all_variants",
    skip_git_check=True,
    variants=S3_SUCCESFUL_VARIANTS,
    )

    mission_variant_curriculum.experiment(
    base_missions=S3_SUCCESFUL_TRAINING_MISSIONS +S3_SUCCESSFUL_EVAL_MISSIONS,
    run_name=f"single_mission_experiment_training_missions_all_variants",
    skip_git_check= True,
    variants=S3_SUCCESFUL_VARIANTS,
    )

def submit_full_curriculum_experiment_standard():
    mission_variant_curriculum.experiment(
    base_missions=None,
    run_name=f"full_curriculum_experiment_standard",
    skip_git_check=True,
    variants=None,
    )

def submit_full_curriculum_experiment_all_variants():
    mission_variant_curriculum.experiment(
    base_missions=None,
    run_name=f"full_curriculum_experiment_all_variants.12.01",
    skip_git_check=True,
    variants="all",
    )

def submit_full_curriculum_with_navigation():
    """Submit full CVC curriculum + Navigation missions (all variants)."""
    # None means full curriculum missions
    # We can't easily append to None, so we fetch the full list
    from recipes.experiment.cvc.mission_variant_curriculum import FULL_CURRICULUM_MISSIONS

    nav_mission_names = [m.name for m in NAVIGATION_MISSIONS]
    combined_missions = list(FULL_CURRICULUM_MISSIONS) + nav_mission_names

    mission_variant_curriculum.experiment(
        base_missions=combined_missions,
        run_name="allmissions_exp_with_nav_no_variants.12.01",
        skip_git_check=True,
        variants=None,
    )

def submit_full_curriculum_with_navigation_with_variants():
    """Submit full CVC curriculum + Navigation missions (all variants)."""
    # None means full curriculum missions
    # We can't easily append to None, so we fetch the full list
    from recipes.experiment.cvc.mission_variant_curriculum import FULL_CURRICULUM_MISSIONS

    nav_mission_names = [m.name for m in NAVIGATION_MISSIONS]
    combined_missions = list(FULL_CURRICULUM_MISSIONS) + nav_mission_names

    mission_variant_curriculum.experiment(
        base_missions=combined_missions,
        run_name="allmissions_experiment_with_nav_all_variants.12.01",
        skip_git_check=True,
        variants="all",
    )

def submit_missions_with_navigation():
    """Submit full CVC curriculum + Navigation missions (all variants)."""
    # None means full curriculum missions
    # We can't easily append to None, so we fetch the full list
    from recipes.experiment.cvc.mission_variant_curriculum import FULL_CURRICULUM_MISSIONS

    nav_mission_names = [m.name for m in NAVIGATION_MISSIONS]
    combined_missions = S3_SUCCESFUL_TRAINING_MISSIONS + S3_SUCCESSFUL_EVAL_MISSIONS + nav_mission_names

    mission_variant_curriculum.experiment(
        base_missions=combined_missions,
        run_name="subset_missions_experiment_with_nav_no_variants.12.01",
        skip_git_check=True,
        variants=None,
    )

def submit_missions_with_navigation_with_variants():
    """Submit full CVC curriculum + Navigation missions (all variants)."""
    # None means full curriculum missions
    # We can't easily append to None, so we fetch the full list
    from recipes.experiment.cvc.mission_variant_curriculum import FULL_CURRICULUM_MISSIONS

    nav_mission_names = [m.name for m in NAVIGATION_MISSIONS]
    combined_missions = S3_SUCCESFUL_TRAINING_MISSIONS + S3_SUCCESSFUL_EVAL_MISSIONS + nav_mission_names

    mission_variant_curriculum.experiment(
        base_missions=combined_missions,
        run_name="subset_missions_experiment_with_nav_all_variants.12.01",
        skip_git_check=True,
        variants="all",
    )

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
        run_name="variants_curriculum_s3_successful_missions_all_variants",
        skip_git_check=True,
        variants="all",  # Curriculum over all variants
    )

    print("\n✓ S3 successful missions curriculum experiment submitted!")


if __name__ == "__main__":
    # submit_variants_experiments()
    # submit_full_curriculum_experiment()
    # submit_single_mission_experiments()
    # submit_all_missions_no_variants()
    # submit_all_missions_with_variants()
    # submit_full_curriculum_experiment_standard()
    submit_full_curriculum_experiment_all_variants()

    # submit_full_curriculum_with_navigation()
    submit_full_curriculum_with_navigation_with_variants()
    # submit_missions_with_navigation()
    # submit_missions_with_navigation_with_variants()

    print("\n" + "=" * 80)
    print("All experiments submitted successfully!")
    print("=" * 80)
