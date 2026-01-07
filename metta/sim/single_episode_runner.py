import json
import logging
import os
import subprocess
import sys
import tempfile
import uuid
from typing import Optional
from uuid import UUID

from opentelemetry import trace as otel_trace
from opentelemetry.trace import SpanKind
from opentelemetry.trace.status import Status, StatusCode
from pydantic import BaseModel

from metta.app_backend.clients.stats_client import StatsClient
from metta.app_backend.models.job_request import JobRequestUpdate
from metta.common.auth.auth_config_reader_writer import observatory_auth_config
from metta.common.otel.tracing import init_tracing, trace_from_carrier
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
    trace_context: Optional[dict[str, str]] = None


logger = logging.getLogger(__name__)
tracer = otel_trace.get_tracer(__name__)


@trace_from_carrier(
    "tournament.job.run",
    carrier_getter=lambda _job_id, job, _stats_client: job.trace_context,
    kind=SpanKind.CONSUMER,
)
def _run_episode(job_id: UUID, job: SingleEpisodeJob, stats_client: StatsClient) -> None:
    span = otel_trace.get_current_span()
    if span.is_recording():
        span.set_attribute("job.id", str(job_id))
        span.set_attribute("job.type", "episode")
        span.set_attribute("job.policy.count", len(job.policy_uris))
        span.set_attribute("job.assignment.count", len(job.assignments))
        span.set_attribute("job.seed", job.seed)
        span.set_attribute("job.max_action_time_ms", job.max_action_time_ms)
        if job.results_uri:
            span.set_attribute("job.results_uri", job.results_uri)
        if job.replay_uri:
            span.set_attribute("job.replay_uri", job.replay_uri)

    local_results_uri = "file://results.json"
    local_replay_uri = "file://replay.json.z"

    try:
        with tracer.start_as_current_span("tournament.job.step.run_simulation") as step_span:
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
                step_span.set_attribute("process.returncode", result.returncode)
                if result.returncode != 0:
                    if result.returncode < 0:
                        # Killed by signal (e.g., OOMKilled sends SIGKILL=-9)
                        signal_num = -result.returncode
                        raise RuntimeError(f"Killed by signal {signal_num}")
                    error_output = result.stderr or result.stdout or "No output"
                    if len(error_output) > 200000:
                        error_output = error_output[:200000] + "\n... (truncated)"
                    raise RuntimeError(
                        f"pure_single_episode_runner failed (exit {result.returncode}):\n{error_output}"
                    )

        with tracer.start_as_current_span("tournament.job.step.upload_results"):
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

        with tracer.start_as_current_span("tournament.job.step.write_episode"):
            episode_tags = {"job_id": str(job_id), **job.episode_tags}
            episode_id = write_single_episode_to_observatory(
                replay_uri=job.replay_uri,
                assignments=job.assignments,
                episode_tags=episode_tags,
                policy_version_ids=policy_version_ids,
                results=results,
                stats_client=stats_client,
            )

        with tracer.start_as_current_span("tournament.job.step.update_job"):
            stats_client.update_job(job_id, JobRequestUpdate(result={"episode_id": str(episode_id)}))

        span = otel_trace.get_current_span()
        if span.is_recording():
            span.set_attribute("episode.id", str(episode_id))
            span.set_attribute("job.outcome", "success")
            span.set_status(Status(StatusCode.OK))
        logger.info(f"Completed job {job_id}")
    except Exception as e:
        span = otel_trace.get_current_span()
        if span.is_recording():
            span.record_exception(e)
            span.set_attribute("job.outcome", "failure")
            span.set_status(Status(StatusCode.ERROR, str(e)))
        raise


def run_episode(job_id: UUID) -> None:
    observatory_auth_config.save_token(os.environ["MACHINE_TOKEN"], os.environ["STATS_SERVER_URI"])

    stats_client = StatsClient.create(os.environ["STATS_SERVER_URI"])

    try:
        job_data = stats_client.get_job(job_id)
        logger.info(f"Started job {job_id}")

        job = SingleEpisodeJob.model_validate(job_data.job)
        _run_episode(job_id, job, stats_client)
    except Exception as e:
        logger.exception(f"Job {job_id} failed")
        stats_client.update_job(job_id, JobRequestUpdate(result={"error": str(e)}))
        raise
    finally:
        stats_client.close()


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m metta.sim.single_episode_runner <job_id>")
        sys.exit(1)

    job_id = UUID(sys.argv[1])
    run_episode(job_id)


if __name__ == "__main__":
    init_logging()
    suppress_noisy_logs()
    init_tracing(service_name="episode-runner")
    main()
