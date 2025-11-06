"""Test that acceptance criteria work correctly at the job level."""

import devops.stable.jobs
import metta.jobs.job_config


def test_acceptance_criteria_in_job_config():
    """Test that acceptance criteria are properly converted to AcceptanceCriterion in JobConfig."""
    job_config = devops.stable.jobs.tool_job(
        name="test_train",
        tool_path="arena.train",
        args=["run=test"],
        acceptance_criteria=[
            metta.jobs.job_config.AcceptanceCriterion(metric="overview/sps", operator=">=", threshold=40000),
            metta.jobs.job_config.AcceptanceCriterion(metric="env_agent/heart.gained", operator=">", threshold=0.5),
        ],
    )

    # Check that JobConfig has AcceptanceCriterion objects
    assert job_config.acceptance_criteria is not None
    assert len(job_config.acceptance_criteria) == 2

    # Verify first criterion
    criterion1 = job_config.acceptance_criteria[0]
    assert isinstance(criterion1, metta.jobs.job_config.AcceptanceCriterion)
    assert criterion1.metric == "overview/sps"
    assert criterion1.operator == ">="
    assert criterion1.threshold == 40000

    # Verify second criterion
    criterion2 = job_config.acceptance_criteria[1]
    assert isinstance(criterion2, metta.jobs.job_config.AcceptanceCriterion)
    assert criterion2.metric == "env_agent/heart.gained"
    assert criterion2.operator == ">"
    assert criterion2.threshold == 0.5

    # Verify metrics_to_track is derived from acceptance
    assert job_config.metrics_to_track == ["overview/sps", "env_agent/heart.gained"]
