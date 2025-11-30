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
    run_name=f"full_curriculum_experiment_all_variants",
    skip_git_check=True,
    variants="all",
    )


if __name__ == "__main__":
    # submit_variants_experiments()
    # submit_full_curriculum_experiment()
    # submit_single_mission_experiments()
    # submit_all_missions_no_variants()
    # submit_all_missions_with_variants()
    submit_full_curriculum_experiment_standard()
    submit_full_curriculum_experiment_all_variants()
    print("\n" + "=" * 80)
    print("All experiments submitted successfully!")
    print("=" * 80)
