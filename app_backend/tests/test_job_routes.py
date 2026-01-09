from metta.app_backend.clients.stats_client import StatsClient
from metta.app_backend.models.job_request import JobRequestCreate, JobRequestUpdate, JobStatus, JobType


class TestEpisodeJobRoutes:
    def test_create_get_update_jobs(self, stats_client: StatsClient):
        job_1 = JobRequestCreate(
            job_type=JobType.episode,
            job={
                "policy_uris": ["file:///tmp/policy1", "file:///tmp/policy2"],
                "assignments": [0, 1, 0, 1],
                "env": {"game": {"num_agents": 4}},
                "results_uri": None,
                "replay_uri": None,
                "seed": 42,
                "max_action_time_ms": 5000,
            },
        )
        job_2 = job_1.model_copy(update={"job": {"policy_uris": ["file:///tmp/policy3", "file:///tmp/policy4"]}})

        jobs = [job_1, job_2]

        # Create teh jobs
        job_ids = stats_client.create_jobs(jobs)

        # Check the job ids are distinct
        assert len(job_ids) == 2
        assert len(set(job_ids)) == 2

        # Fetch them by id and check that they match what we submitted
        for i, j in enumerate(jobs):
            job = stats_client.get_job(job_ids[i])
            assert job.job_type == JobType.episode
            assert job.id == job_ids[i]
            assert job.job == j.job

        # Jobs start as dispatched (dispatch_job stub succeeds)
        fetched_job = stats_client.get_job(job_ids[0])
        assert fetched_job.status == JobStatus.dispatched

        # Update to running
        stats_client.update_job(job_ids[0], JobRequestUpdate(status=JobStatus.running, worker="worker1"))
        fetched_job = stats_client.get_job(job_ids[0])
        assert fetched_job.status == JobStatus.running

        # Update to completed
        stats_client.update_job(job_ids[0], JobRequestUpdate(status=JobStatus.completed))
        fetched_job = stats_client.get_job(job_ids[0])
        assert fetched_job.status == JobStatus.completed

        # Fetch jobs by status and confirm filtering works
        completed_jobs = stats_client.list_jobs(statuses=[JobStatus.completed])
        assert len(completed_jobs) == 1
        assert completed_jobs[0].id == job_ids[0]

        dispatched_jobs = stats_client.list_jobs(statuses=[JobStatus.dispatched])
        assert len(dispatched_jobs) == 1
        assert dispatched_jobs[0].id == job_ids[1]
