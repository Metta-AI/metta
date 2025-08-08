"""
Simple test example for the A/B testing framework.
"""

from metta.ab_test.config import create_experiment


def create_test_experiment():
    """Create a simple test experiment."""
    return (
        create_experiment(name="test_experiment", description="Simple test experiment")
        .add_variant(name="variant_a", description="First test variant", run="test_a")
        .add_variant(name="variant_b", description="Second test variant", run="test_b")
        .set_runs_per_variant(2)
        .set_base_config(
            defaults=["/common"],
            total_timesteps=1000,  # Very short for testing
            trainer__num_workers=1,
        )
        .build()
    )


# For direct execution
if __name__ == "__main__":
    experiment = create_test_experiment()
    print(f"Test experiment: {experiment.name}")
    print(f"Variants: {list(experiment.variants.keys())}")
    print(f"Runs per variant: {experiment.runs_per_variant}")
