"""Test that acceptance criteria are displayed correctly."""

from operator import ge, gt

from devops.stable.tasks import tool_task
from metta.jobs.job_config import JobConfig
from metta.jobs.job_manager import JobManager


def test_acceptance_criteria_in_config(tmp_path):
    """Test that acceptance criteria are properly set in JobConfig."""
    task = tool_task(
        name="test_train",
        module="arena.train",
        args=["run=test"],
        acceptance=[
            ("overview/sps", ge, 40000),
            ("env_agent/heart.gained", gt, 0.5),
        ],
    )

    # Check that acceptance criteria are set
    assert task.job_config.acceptance_criteria is not None
    assert "overview/sps" in task.job_config.acceptance_criteria
    assert "env_agent/heart.gained" in task.job_config.acceptance_criteria

    # Check the values
    assert task.job_config.acceptance_criteria["overview/sps"] == (">=", 40000)
    assert task.job_config.acceptance_criteria["env_agent/heart.gained"] == (">", 0.5)


def test_acceptance_criteria_persisted(tmp_path):
    """Test that acceptance criteria survive database round-trip."""
    job_manager = JobManager(tmp_path)

    config = JobConfig(
        name="test_job",
        module="test.module",
        is_training_job=True,
        metrics_to_track=["overview/sps"],
        acceptance_criteria={"overview/sps": (">=", 40000)},
    )

    # Submit and immediately retrieve
    job_manager.submit(config)
    job_state = job_manager.get_job_state("test_job")

    # Check that acceptance_criteria was persisted
    assert job_state is not None
    assert job_state.config.acceptance_criteria is not None
    assert "overview/sps" in job_state.config.acceptance_criteria
    assert job_state.config.acceptance_criteria["overview/sps"] == (">=", 40000)
