"""Tests for database layer."""

import os
import tempfile

import pytest
from skydeck.database import Database
from skydeck.models import DesiredState, Experiment, Job, JobStatus


@pytest.fixture
async def db():
    """Create a temporary database for testing."""
    # Create temp file
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)

    database = Database(path)
    await database.connect()

    yield database

    await database.close()
    os.unlink(path)


@pytest.mark.asyncio
async def test_save_and_get_experiment(db):
    """Test saving and retrieving an experiment."""
    experiment = Experiment(
        id="test_exp",
        name="Test Experiment",
        desired_state=DesiredState.RUNNING,
        current_state=JobStatus.INIT,
        flags={"key": "value"},
    )

    await db.save_experiment(experiment)

    retrieved = await db.get_experiment("test_exp")
    assert retrieved is not None
    assert retrieved.id == "test_exp"
    assert retrieved.name == "Test Experiment"
    assert retrieved.desired_state == DesiredState.RUNNING
    assert retrieved.flags == {"key": "value"}


@pytest.mark.asyncio
async def test_get_all_experiments(db):
    """Test retrieving all experiments."""
    exp1 = Experiment(id="exp1", name="Exp 1")
    exp2 = Experiment(id="exp2", name="Exp 2")

    await db.save_experiment(exp1)
    await db.save_experiment(exp2)

    all_exps = await db.get_all_experiments()
    assert len(all_exps) == 2
    assert {e.id for e in all_exps} == {"exp1", "exp2"}


@pytest.mark.asyncio
async def test_delete_experiment(db):
    """Test deleting an experiment."""
    experiment = Experiment(id="test_exp", name="Test")
    await db.save_experiment(experiment)

    await db.delete_experiment("test_exp")

    retrieved = await db.get_experiment("test_exp")
    assert retrieved is None


@pytest.mark.asyncio
async def test_save_and_get_job(db):
    """Test saving and retrieving a job."""
    # First create an experiment
    experiment = Experiment(id="test_exp", name="Test")
    await db.save_experiment(experiment)

    # Create a job
    job = Job(
        id="job1",
        experiment_id="test_exp",
        cluster_name="cluster1",
        command="test command",
        status=JobStatus.RUNNING,
    )

    await db.save_job(job)

    retrieved = await db.get_job("job1")
    assert retrieved is not None
    assert retrieved.id == "job1"
    assert retrieved.experiment_id == "test_exp"
    assert retrieved.status == JobStatus.RUNNING


@pytest.mark.asyncio
async def test_get_jobs_for_experiment(db):
    """Test getting jobs for an experiment."""
    # Create experiment
    experiment = Experiment(id="test_exp", name="Test")
    await db.save_experiment(experiment)

    # Create multiple jobs
    for i in range(5):
        job = Job(
            id=f"job{i}",
            experiment_id="test_exp",
            cluster_name="cluster1",
            command="test",
            status=JobStatus.RUNNING,
        )
        await db.save_job(job)

    jobs = await db.get_jobs_for_experiment("test_exp", limit=3)
    assert len(jobs) == 3


@pytest.mark.asyncio
async def test_update_experiment_state(db):
    """Test updating experiment state."""
    experiment = Experiment(
        id="test_exp",
        name="Test",
        current_state=JobStatus.INIT,
    )
    await db.save_experiment(experiment)

    await db.update_experiment_state("test_exp", JobStatus.RUNNING, "job1")

    retrieved = await db.get_experiment("test_exp")
    assert retrieved.current_state == JobStatus.RUNNING
    assert retrieved.current_job_id == "job1"
