#!/usr/bin/env python
"""Submit test episode jobs to the local backend."""

import argparse

from metta.app_backend.clients.stats_client import StatsClient
from metta.app_backend.models.job_request import JobRequestCreate, JobType
from mettagrid import MettaGridConfig


def main():
    parser = argparse.ArgumentParser(description="Submit test episode jobs")
    parser.add_argument("--server", default="http://localhost:8000", help="Backend URL")
    parser.add_argument("--num-agents", type=int, default=2, help="Number of agents")
    parser.add_argument("--num-jobs", type=int, default=1, help="Number of jobs to submit")
    parser.add_argument("--policy-uri", help="Policy URI")
    args = parser.parse_args()

    if not args.policy_uri:
        raise ValueError("Policy URI is required")

    policy_uris = [args.policy_uri]

    env = MettaGridConfig.EmptyRoom(num_agents=args.num_agents, width=20, height=20)

    jobs = []
    for seed in range(args.num_jobs):
        job = {
            "policy_uris": policy_uris,
            "assignments": [i % len(policy_uris) for i in range(args.num_agents)],
            "env": env.model_dump(),
            "results_uri": None,
            "replay_uri": None,
            "seed": seed,
        }
        jobs.append(JobRequestCreate(job_type=JobType.episode, job=job))

    stats_client = StatsClient(args.server)
    try:
        job_ids = stats_client.create_jobs(jobs)
        print(f"Created {len(job_ids)} job(s):")
        for job_id in job_ids:
            print(f"  {job_id}")
    finally:
        stats_client.close()


if __name__ == "__main__":
    main()
