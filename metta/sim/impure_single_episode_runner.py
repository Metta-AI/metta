import json
import logging
import os
import subprocess
import sys
import uuid
from uuid import UUID

from metta.app_backend.clients.stats_client import StatsClient
from metta.app_backend.models.job_request import JobRequestUpdate, JobStatus
from metta.common.auth.auth_config_reader_writer import observatory_auth_config
from metta.rl.metta_scheme_resolver import MettaSchemeResolver
from metta.sim.handle_results import write_single_episode_to_observatory
from metta.sim.pure_single_episode_runner import PureSingleEpisodeJob, PureSingleEpisodeResult
from mettagrid.policy.mpt_policy import parse_uri
from mettagrid.util.file import copy_data, read

logger = logging.getLogger(__name__)


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m metta.app_backend.job_runner.episode_job_worker <job_id>")
        sys.exit(1)

    job_id = UUID(sys.argv[1])
    observatory_auth_config.save_token(os.environ["MACHINE_TOKEN"], os.environ["BACKEND_URL"])

    worker_name = os.environ["WORKER_NAME"]

    stats_client = StatsClient.create(os.environ["BACKEND_URL"])

    try:
        job_data = stats_client.get_job(job_id)

        stats_client.update_job(job_id, JobRequestUpdate(status=JobStatus.running, worker=worker_name))
        logger.info(f"Started job {job_id} on worker {worker_name}")

        job = PureSingleEpisodeJob.model_validate(job_data.job)

        # Run the episode using the pure-runner in another process
        # Once we make trained policies not require network to load (after initial download),
        # this subprocess should be runnable with no network
        local_results_uri = "file://results.json"
        local_replay_uri = "file://replay.json.z"
        local_job = job.model_copy(deep=True, update={"results_uri": local_results_uri, "replay_uri": local_replay_uri})
        subprocess.run(
            [
                "/workspace/metta/.venv/bin/python",
                "-m",
                "metta.sim.pure_single_episode_runner",
                json.dumps(
                    {
                        "job": local_job.model_dump(),
                        "device": "cpu",
                        "allow_network": True,  # Until trained policies no longer need network access to hydrate
                    }
                ),
            ]
        )

        # Copy local pure-runner's results to requested locations
        for src, dest, content_type in [
            (local_replay_uri, job.replay_uri, "application/x-compress"),
            (local_results_uri, job.results_uri, "application/json"),
        ]:
            if dest is not None:
                copy_data(src, dest, content_type=content_type)

        # Upload results to Observatory database
        results = PureSingleEpisodeResult.model_validate_json(read(local_results_uri))

        policy_version_ids: list[uuid.UUID | None] = []
        backend_url = os.environ["BACKEND_URL"]
        for policy_uri in job.policy_uris:
            parsed = parse_uri(policy_uri, allow_none=False)
            if parsed.scheme != "metta":
                policy_version_ids.append(None)
            else:
                policy_version = MettaSchemeResolver(backend_url).get_policy_version(policy_uri)
                policy_version_ids.append(policy_version.id)

        episode_id = write_single_episode_to_observatory(
            episode_tags={"job_id": str(job_id)},
            policy_version_ids=policy_version_ids,
            job=job,
            results=results,
            stats_client=stats_client,
        )

        # Update job status and result
        stats_client.update_job(
            job_id, JobRequestUpdate(status=JobStatus.completed, result={"episode_id": str(episode_id)})
        )
        logger.info(f"Completed job {job_id}")

    except Exception as e:
        logger.exception(f"Job {job_id} failed")
        stats_client.update_job(job_id, JobRequestUpdate(status=JobStatus.failed, error=str(e)))
        raise
    finally:
        stats_client.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
