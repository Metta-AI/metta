"""Example experiment demonstrating the experiment system.

This shows how to define and run an experiment with multiple jobs.
"""

from metta.experiment.tool import ExperimentTool
from metta.jobs.models import JobSpec


def simple_example() -> ExperimentTool:
    """Simple example: train two policies with different learning rates.

    Usage:
        # Launch experiment
        ./tools/run.py experiments.user.example_experiment.simple_example

        # Monitor progress
        ./tools/run.py experiments.user.example_experiment.simple_example mode=monitor

        # Attach to running job
        ./tools/run.py experiments.user.example_experiment.simple_example mode=attach

        # Cancel all jobs
        ./tools/run.py experiments.user.example_experiment.simple_example mode=cancel
    """
    jobs = [
        JobSpec(
            name="lr_0001",
            module="experiments.recipes.arena.train",
            args={"run": "example_exp.lr_0001"},
            overrides={
                "trainer.optimizer.learning_rate": 0.0001,
                "trainer.total_timesteps": 10_000_000,
            },
            execution="remote",
            gpus=1,
            nodes=1,
            spot=True,
            timeout_s=7200,
        ),
        JobSpec(
            name="lr_0003",
            module="experiments.recipes.arena.train",
            args={"run": "example_exp.lr_0003"},
            overrides={
                "trainer.optimizer.learning_rate": 0.0003,
                "trainer.total_timesteps": 10_000_000,
            },
            execution="remote",
            gpus=1,
            nodes=1,
            spot=True,
            timeout_s=7200,
        ),
    ]

    return ExperimentTool(
        name="simple_example",
        jobs=jobs,
    )


def ab_test_example() -> ExperimentTool:
    """A/B test example: compare baseline vs shaped rewards.

    This demonstrates a more complex experiment with different recipes.
    """
    jobs = [
        # Baseline: standard arena training
        JobSpec(
            name="baseline",
            module="experiments.recipes.arena.train",
            args={"run": "ab_test.baseline"},
            overrides={"trainer.total_timesteps": 50_000_000},
            execution="remote",
            gpus=4,
            nodes=1,
        ),
        # Treatment: shaped rewards recipe
        JobSpec(
            name="shaped_rewards",
            module="experiments.recipes.arena_basic_easy_shaped.train",
            args={"run": "ab_test.shaped"},
            overrides={"trainer.total_timesteps": 50_000_000},
            execution="remote",
            gpus=4,
            nodes=1,
        ),
    ]

    return ExperimentTool(
        name="ab_test_example",
        jobs=jobs,
    )


def parameter_sweep() -> ExperimentTool:
    """Parameter sweep example: test multiple hyperparameter combinations.

    This shows how to programmatically generate jobs for a parameter sweep.
    """
    learning_rates = [0.0001, 0.0003, 0.001]
    batch_sizes = [256, 512]

    jobs = []

    for lr in learning_rates:
        for batch_size in batch_sizes:
            # Generate descriptive name
            lr_str = f"{int(lr * 10000):04d}"
            name = f"lr{lr_str}_bs{batch_size}"

            jobs.append(
                JobSpec(
                    name=name,
                    module="experiments.recipes.arena.train",
                    args={"run": f"param_sweep.{name}"},
                    overrides={
                        "trainer.optimizer.learning_rate": lr,
                        "trainer.batch_size": batch_size,
                        "trainer.total_timesteps": 20_000_000,
                    },
                    execution="remote",
                    gpus=2,
                    metadata={
                        "learning_rate": lr,
                        "batch_size": batch_size,
                    },
                )
            )

    return ExperimentTool(
        name="parameter_sweep",
        jobs=jobs,
    )


def local_test_example() -> ExperimentTool:
    """Local execution example: run quick test jobs locally.

    This demonstrates using execution="local" for testing without SkyPilot.
    """
    jobs = [
        JobSpec(
            name="quick_test_1",
            module="experiments.recipes.arena.train",
            args={"run": "local_test.job1"},
            overrides={"trainer.total_timesteps": 100_000},
            execution="local",  # Run locally instead of remote
            timeout_s=600,  # 10 minutes
        ),
        JobSpec(
            name="quick_test_2",
            module="experiments.recipes.arena.train",
            args={"run": "local_test.job2"},
            overrides={"trainer.total_timesteps": 100_000},
            execution="local",
            timeout_s=600,
        ),
    ]

    return ExperimentTool(
        name="local_test_example",
        jobs=jobs,
    )
