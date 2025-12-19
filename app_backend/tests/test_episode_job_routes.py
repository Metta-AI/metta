from metta.app_backend.clients.stats_client import StatsClient


class TestEpisodeJobRoutes:
    def test_create_multiple_jobs(self, stats_client: StatsClient):
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

        job_ids = stats_client.create_episode_jobs(jobs)

        assert len(job_ids) == 2
        assert len(set(job_ids)) == 2

        for i, j in enumerate(jobs):
            job = stats_client.get_episode_job(job_ids[i])
            assert job.id == job_ids[i]
            assert job.job == j
