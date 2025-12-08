"""Tests for data models."""

from skydeck.models import (
    DesiredState,
    Experiment,
    Job,
    JobStatus,
)


def test_experiment_build_command():
    """Test experiment command building."""
    experiment = Experiment(
        id="test",
        name="Test",
        base_command="lt",
        run_name="daveey.test",
        flags={
            "trainer.losses.ppo.enabled": True,
            "trainer.total_timesteps": 100000,
            "policy_architecture.core_resnet_layers": 4,
        },
    )

    command = experiment.build_command()
    assert "lt" in command
    assert "run=daveey.test" in command
    assert "trainer.losses.ppo.enabled=true" in command
    assert "trainer.total_timesteps=100000" in command
    assert "policy_architecture.core_resnet_layers=4" in command


def test_experiment_needs_reconciliation():
    """Test reconciliation detection."""
    # RUNNING desired, INIT current -> needs reconciliation
    exp1 = Experiment(
        id="test1",
        name="Test 1",
        desired_state=DesiredState.RUNNING,
        current_state=JobStatus.INIT,
    )
    assert exp1.needs_reconciliation()

    # RUNNING desired, RUNNING current -> no reconciliation
    exp2 = Experiment(
        id="test2",
        name="Test 2",
        desired_state=DesiredState.RUNNING,
        current_state=JobStatus.RUNNING,
    )
    assert not exp2.needs_reconciliation()

    # STOPPED desired, RUNNING current -> needs reconciliation
    exp3 = Experiment(
        id="test3",
        name="Test 3",
        desired_state=DesiredState.STOPPED,
        current_state=JobStatus.RUNNING,
    )
    assert exp3.needs_reconciliation()


def test_job_is_terminal():
    """Test job terminal state detection."""
    job_succeeded = Job(
        id="job1",
        experiment_id="exp1",
        cluster_name="cluster1",
        command="test",
        status=JobStatus.SUCCEEDED,
    )
    assert job_succeeded.is_terminal()

    job_running = Job(
        id="job2",
        experiment_id="exp1",
        cluster_name="cluster1",
        command="test",
        status=JobStatus.RUNNING,
    )
    assert not job_running.is_terminal()


def test_job_is_active():
    """Test job active state detection."""
    job_running = Job(
        id="job1",
        experiment_id="exp1",
        cluster_name="cluster1",
        command="test",
        status=JobStatus.RUNNING,
    )
    assert job_running.is_active()

    job_pending = Job(
        id="job2",
        experiment_id="exp1",
        cluster_name="cluster1",
        command="test",
        status=JobStatus.PENDING,
    )
    assert job_pending.is_active()

    job_failed = Job(
        id="job3",
        experiment_id="exp1",
        cluster_name="cluster1",
        command="test",
        status=JobStatus.FAILED,
    )
    assert not job_failed.is_active()
