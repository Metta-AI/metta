"""Test that acceptance criteria work correctly at the job level."""

from metta.jobs.job_config import AcceptanceCriterion, JobConfig


def test_acceptance_criteria_in_job_config():
    """Test that acceptance criteria are properly stored in JobConfig."""
    job_config = JobConfig(
        name="test_train",
        module="recipes.prod.arena_basic_easy_shaped.train",
        args=["run=test"],
        acceptance_criteria=[
            AcceptanceCriterion(metric="overview/sps", operator=">=", threshold=40000),
            AcceptanceCriterion(metric="env_game/assembler.heart.created", operator=">", threshold=0.5),
        ],
    )

    # Check that JobConfig has AcceptanceCriterion objects
    assert job_config.acceptance_criteria is not None
    assert len(job_config.acceptance_criteria) == 2

    # Verify first criterion
    criterion1 = job_config.acceptance_criteria[0]
    assert isinstance(criterion1, AcceptanceCriterion)
    assert criterion1.metric == "overview/sps"
    assert criterion1.operator == ">="
    assert criterion1.threshold == 40000

    # Verify second criterion
    criterion2 = job_config.acceptance_criteria[1]
    assert isinstance(criterion2, AcceptanceCriterion)
    assert criterion2.metric == "env_game/assembler.heart.created"
    assert criterion2.operator == ">"
    assert criterion2.threshold == 0.5
