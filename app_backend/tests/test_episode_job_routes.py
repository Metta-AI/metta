from metta.app_backend.clients.stats_client import StatsClient
from metta.app_backend.models.job_request import JobStatus


class TestEpisodeJobRoutes:
    def test_create_get_update_jobs(self, stats_client: StatsClient):
        job_1 = {
            "policy_uris": ["file:///tmp/policy1", "file:///tmp/policy2"],
            "assignments": [0, 1, 0, 1],
            "env": {"game": {"num_agents": 4}},
            "results_uri": None,
            "replay_uri": None,
            "seed": 42,
            "max_action_time_ms": 5000,
        }
        job_2 = {**job_1, "policy_uris": ["file:///tmp/policy3", "file:///tmp/policy4"]}

        jobs = [job_1, job_2]

        # Create teh jobs
        job_ids = stats_client.create_episode_jobs(jobs)

        # Check the job ids are distinct
        assert len(job_ids) == 2
        assert len(set(job_ids)) == 2

        # Fetch them by id and check that they match what we submitted
        for i, j in enumerate(jobs):
            job = stats_client.get_episode_job(job_ids[i])
            assert job.id == job_ids[i]
            assert job.job == j

        # Update one
        stats_client.update_episode_job(job_ids[0], JobStatus.dispatched)

        # Fetch it again and check that the result is updated
        fetched_job = stats_client.get_episode_job(job_ids[0])
        assert fetched_job.status == JobStatus.dispatched

        # Fetch jobs by status and confirm it's in there
        jobs = stats_client.list_episode_jobs(status=JobStatus.dispatched)
        assert len(jobs) == 1
        assert jobs[0].id == job_ids[0]
