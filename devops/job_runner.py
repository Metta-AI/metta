"""Job runner for local and remote execution.

DEPRECATED: This module has been moved to metta.jobs.runner for better code reuse.
All classes are re-exported here for backwards compatibility.

Provides a unified interface for running jobs locally via subprocess
or remotely via SkyPilot, with support for both async and sync execution.

Example usage:
    # Local job (sync)
    job = LocalJob(name="test", cmd=["pytest", "tests/"], timeout_s=900)
    result = job.wait(stream_output=True)
    print(f"Exit code: {result.exit_code}")

    # Remote job (sync)
    job = RemoteJob(
        name="train",
        module="experiments.recipes.arena.train",
        args=["run=test", "trainer.total_timesteps=100000"],
        timeout_s=3600,
        base_args=["--gpus=4", "--no-spot"]
    )
    result = job.wait(stream_output=True)
    print(f"Job ID: {result.job_id}, Exit code: {result.exit_code}")

    # Remote job (async - poll for completion)
    job = RemoteJob(
        name="train",
        module="experiments.recipes.arena.train",
        args=["run=test"]
    )
    job.submit()
    while not job.is_complete():
        time.sleep(10)
    result = job.get_result()

    # Attach to existing remote job
    job = RemoteJob(
        name="train",
        module="experiments.recipes.arena.train",
        args=["run=test"],
        job_id=12345
    )
    result = job.wait(stream_output=True)
"""

# Back-compat: re-export from new shared location
from metta.jobs.runner import Job, JobResult, LocalJob, RemoteJob

__all__ = ["Job", "JobResult", "LocalJob", "RemoteJob"]
