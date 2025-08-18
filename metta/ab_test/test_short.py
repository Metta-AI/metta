"""
Very short test experiment to verify training works.
"""

from metta.ab_test.config import create_experiment


def create_short_test_experiment():
    """Create a very short test experiment."""
    return (
        create_experiment(name="short_test", description="Very short test experiment")
        .add_variant(name="variant_a", description="First test variant", run="short_a")
        .add_variant(name="variant_b", description="Second test variant", run="short_b")
        .set_runs_per_variant(1)  # Only 1 run per variant
        .set_base_config(
            defaults=["/common"],
            trainer__total_timesteps=100,  # Very short
            trainer__num_workers=1,
        )
        .build()
    )
