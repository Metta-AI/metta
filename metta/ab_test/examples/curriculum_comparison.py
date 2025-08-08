"""
Example A/B test experiment: Curriculum Comparison

This example demonstrates how to define an A/B test comparing
learning progress curriculum vs prioritized regressed curriculum.
"""

from metta.ab_test.config import create_experiment
from metta.ab_test.runner import run_ab_test


def create_curriculum_comparison_experiment():
    """Create the curriculum comparison A/B test experiment."""

    experiment = (
        create_experiment(
            name="curriculum_comparison", description="Learning Progress vs Prioritized Regressed Curriculum"
        )
        .add_variant(
            name="learning_progress",
            description="Learning progress curriculum for navigation tasks",
            trainer__curriculum="/env/mettagrid/curriculum/navigation/learning_progress",
            run="curriculum_lp",
        )
        .add_variant(
            name="prioritized_regressed",
            description="Prioritized regressed curriculum for navigation tasks",
            trainer__curriculum="/env/mettagrid/curriculum/nav_memory_sequence",
            run="curriculum_pr",
        )
        .set_runs_per_variant(1)  # Temporarily 1 for testing
        .set_base_config(
            defaults=["/common", "/agent/fast", "/trainer/trainer", "/sim/all"],
            trainer__total_timesteps=1_000_000_000,
            trainer__num_workers=4,
            trainer__simulation__evaluate_interval=100,
            trainer__checkpoint__wandb_checkpoint_interval=100,
        )
        .set_wandb_config(entity="metta-research")
        .build()
    )

    return experiment


def run_curriculum_comparison():
    """Run the curriculum comparison experiment."""
    experiment = create_curriculum_comparison_experiment()

    print(f"Running A/B test: {experiment.name}")
    print(f"Description: {experiment.description}")
    print(f"Variants: {list(experiment.variants.keys())}")
    print(f"Runs per variant: {experiment.runs_per_variant}")
    print(f"WandB Project: {experiment.wandb_project}")

    results = run_ab_test(experiment, output_dir="ab_test_results", parallel_runs=False, retry_failed_runs=True)

    return results


if __name__ == "__main__":
    run_curriculum_comparison()
