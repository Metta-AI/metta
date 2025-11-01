"""Test that acceptance criteria work correctly at the job level."""

from operator import ge, gt

from devops.stable.tasks import tool_task
from metta.jobs.job_config import AcceptanceCriterion


def test_acceptance_criteria_in_job_config():
    """Test that acceptance criteria are properly converted to AcceptanceCriterion in JobConfig."""
    task = tool_task(
        name="test_train",
        module="arena.train",
        args=["run=test"],
        acceptance=[
            ("overview/sps", ge, 40000),
            ("env_agent/heart.gained", gt, 0.5),
        ],
    )

    # Check that acceptance criteria are set on both Task and JobConfig
    assert task.acceptance is not None
    assert len(task.acceptance) == 2

    # Check that JobConfig has AcceptanceCriterion objects
    assert task.job_config.acceptance_criteria is not None
    assert len(task.job_config.acceptance_criteria) == 2

    # Verify first criterion
    criterion1 = task.job_config.acceptance_criteria[0]
    assert isinstance(criterion1, AcceptanceCriterion)
    assert criterion1.metric == "overview/sps"
    assert criterion1.operator == ">="
    assert criterion1.threshold == 40000

    # Verify second criterion
    criterion2 = task.job_config.acceptance_criteria[1]
    assert isinstance(criterion2, AcceptanceCriterion)
    assert criterion2.metric == "env_agent/heart.gained"
    assert criterion2.operator == ">"
    assert criterion2.threshold == 0.5

    # Verify metrics_to_track is derived from acceptance
    assert task.job_config.metrics_to_track == ["overview/sps", "env_agent/heart.gained"]
