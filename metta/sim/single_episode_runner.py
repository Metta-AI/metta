import json
import logging
import os
import subprocess
import sys
import tempfile
import uuid
from uuid import UUID

from pydantic import BaseModel

from metta.app_backend.clients.stats_client import StatsClient
from metta.app_backend.models.job_request import JobRequestUpdate
from metta.common.auth.auth_config_reader_writer import observatory_auth_config
from metta.common.util.log_config import init_logging, suppress_noisy_logs
from metta.rl.metta_scheme_resolver import MettaSchemeResolver
from metta.sim.handle_results import write_single_episode_to_observatory
from metta.sim.pure_single_episode_runner import PureSingleEpisodeJob, PureSingleEpisodeResult
from mettagrid import MettaGridConfig
from mettagrid.util.file import copy_data, read
from mettagrid.util.uri_resolvers.schemes import parse_uri


class SingleEpisodeJob(BaseModel):
    policy_uris: list[str]
    assignments: list[int]
    env: MettaGridConfig
    results_uri: str | None = None
    replay_uri: str | None = None
    seed: int = 0
    max_action_time_ms: int = 10000
    episode_tags: dict[str, str] = {}


logger = logging.getLogger(__name__)


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m metta.sim.single_episode_runner <job_id>")
        sys.exit(1)

    job_id = UUID(sys.argv[1])

    observatory_auth_config.save_token(os.environ["MACHINE_TOKEN"], os.environ["STATS_SERVER_URI"])

    stats_client = StatsClient.create(os.environ["STATS_SERVER_URI"])

    try:
        job_data = stats_client.get_job(job_id)
        logger.info(f"Started job {job_id}")

        job = SingleEpisodeJob.model_validate(job_data.job)

        local_results_uri = "file://results.json"
        local_replay_uri = "file://replay.json.z"

        with tempfile.NamedTemporaryFile(delete=True) as temp_file:
            pure_job_spec = {
                "job": PureSingleEpisodeJob(
                    policy_uris=job.policy_uris,
                    assignments=job.assignments,
                    env=job.env,
                    results_uri=local_results_uri,
                    replay_uri=local_replay_uri,
                    seed=job.seed,
                    max_action_time_ms=job.max_action_time_ms,
                ).model_dump(),
                "device": "cpu",
                "allow_network": True,
            }
            temp_file.write(json.dumps(pure_job_spec).encode("utf-8"))
            temp_file.flush()
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "metta.sim.pure_single_episode_runner",
                    temp_file.name,
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                error_output = result.stderr or result.stdout or "No output"
                if len(error_output) > 2000:
                    error_output = error_output[:2000] + "\n... (truncated)"
                raise RuntimeError(f"pure_single_episode_runner failed (exit {result.returncode}):\n{error_output}")

        for src, dest, content_type in [
            (local_replay_uri, job.replay_uri, "application/x-compress"),
            (local_results_uri, job.results_uri, "application/json"),
        ]:
            if dest is not None:
                copy_data(src, dest, content_type=content_type)

        results = PureSingleEpisodeResult.model_validate_json(read(local_results_uri))

        policy_version_ids: list[uuid.UUID | None] = []
        stats_server_uri = os.environ["STATS_SERVER_URI"]
        for policy_uri in job.policy_uris:
            parsed = parse_uri(policy_uri, allow_none=False)
            if parsed.scheme != "metta":
                policy_version_ids.append(None)
            else:
                policy_version = MettaSchemeResolver(stats_server_uri).get_policy_version(policy_uri)
                policy_version_ids.append(policy_version.id)

        episode_tags = {"job_id": str(job_id), **job.episode_tags}
        episode_id = write_single_episode_to_observatory(
            replay_uri=job.replay_uri,
            assignments=job.assignments,
            episode_tags=episode_tags,
            policy_version_ids=policy_version_ids,
            results=results,
            stats_client=stats_client,
        )

        stats_client.update_job(job_id, JobRequestUpdate(result={"episode_id": str(episode_id)}))
        logger.info(f"Completed job {job_id}")

    except Exception as e:
        logger.exception(f"Job {job_id} failed")
        stats_client.update_job(job_id, JobRequestUpdate(result={"error": str(e)}))
        raise
    finally:
        stats_client.close()


if __name__ == "__main__":
    init_logging()
    suppress_noisy_logs()
    main()
