"""Experiment system for managing groups of related training/evaluation jobs.

This package provides infrastructure for launching and monitoring experiments
consisting of 2-20 related jobs (e.g., parameter variations, A/B tests).

Key components:
- tool.py: ExperimentTool for defining experiments
- state.py: State persistence and management
- launcher.py: Job launch logic
- monitor.py: Live monitoring and log streaming
- manager.py: Experiment lifecycle management

Usage:
    def my_experiment() -> ExperimentTool:
        return ExperimentTool(
            name="my_experiment",
            jobs=[
                JobSpec(
                    name="job1",
                    module="experiments.recipes.arena.train",
                    args={"run": "experiment.job1"},
                ),
            ],
        )
"""

from metta.experiment.state import ExperimentState, JobState

__all__ = ["ExperimentState", "JobState"]
