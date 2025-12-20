import logging
from uuid import UUID

import kopf
from kubernetes import client
from kubernetes import config as kubernetes_config

from metta.app_backend.clients.stats_client import StatsClient
from metta.app_backend.job_runner.config import get_job_runner_config
from metta.app_backend.models.job_request import JobRequestUpdate, JobStatus, JobType
from metta.common.auth.auth_config_reader_writer import observatory_auth_config
from metta.common.util.log_config import init_logging, suppress_noisy_logs

logger = logging.getLogger(__name__)


def create_k8s_job(
    batch_v1: client.BatchV1Api,
    job_id: UUID,
) -> None:
    env_config = get_job_runner_config()
    job = client.V1Job(
        metadata=client.V1ObjectMeta(
            name=f"job-{job_id.hex[:8]}",
            namespace=env_config.KUBERNETES_NAMESPACE,
            labels={
                "app": "episode-runner",
                "job-id": str(job_id),
            },
        ),
        spec=client.V1JobSpec(
            backoff_limit=3,
            active_deadline_seconds=3600,
            ttl_seconds_after_finished=300,
            template=client.V1PodTemplateSpec(
                metadata=client.V1ObjectMeta(
                    labels={
                        "app": "episode-runner",
                        "job-id": str(job_id),
                    },
                ),
                spec=client.V1PodSpec(
                    restart_policy="Never",
                    containers=[
                        client.V1Container(
                            name="runner",
                            image=env_config.DOCKER_IMAGE,
                            command=[
                                "python",
                                "-m",
                                "metta.app_backend.job_runner.episode_job_worker",
                                str(job_id),
                            ],
                            env=[  # Pass along all env vars
                                client.V1EnvVar(name=name, value=value)
                                for name, value in env_config.model_dump().items()
                            ],
                            resources=client.V1ResourceRequirements(
                                requests={"cpu": "3", "memory": "3Gi"},
                                limits={"cpu": "4", "memory": "6Gi"},
                            ),
                        )
                    ],
                ),
            ),
        ),
    )
    batch_v1.create_namespaced_job(namespace=env_config.KUBERNETES_NAMESPACE, body=job)
    logger.info(f"Created k8s Job for job {job_id}")


stats_client: StatsClient | None = None


@kopf.on.startup()  # type: ignore[arg-type]
def configure(settings: kopf.OperatorSettings, **_):
    global stats_client
    settings.posting.level = logging.WARNING

    try:
        kubernetes_config.load_incluster_config()
    except kubernetes_config.ConfigException:
        kubernetes_config.load_kube_config()

    env_config = get_job_runner_config()
    observatory_auth_config.save_token(env_config.MACHINE_TOKEN, env_config.BACKEND_URL)

    stats_client = StatsClient(env_config.BACKEND_URL)
    logger.info(f"Job runner started with backend_url={env_config.BACKEND_URL}")


@kopf.on.cleanup()  # type: ignore[arg-type]
def cleanup(**_):
    global stats_client
    if stats_client:
        stats_client.close()
        stats_client = None


@kopf.timer("", interval=5.0, initial_delay=1.0)  # type: ignore[arg-type]
def reconcile_jobs(**_):
    if not stats_client:
        logger.error("Stats client not initialized")
        return

    env_config = get_job_runner_config()
    batch_v1 = client.BatchV1Api()

    pending_jobs = stats_client.list_jobs(job_type=JobType.episode, statuses=[JobStatus.pending], limit=50)
    pending_ids = {j.id for j in pending_jobs}

    k8s_jobs = batch_v1.list_namespaced_job(
        namespace=env_config.KUBERNETES_NAMESPACE, label_selector="app=episode-runner"
    )
    existing_job_ids = {job.metadata.labels.get("job-id") for job in k8s_jobs.items if job.metadata.labels}

    for job_id in pending_ids:
        if str(job_id) not in existing_job_ids:
            try:
                create_k8s_job(batch_v1, job_id)
                stats_client.update_job(job_id, JobRequestUpdate(status=JobStatus.dispatched))
                logger.info(f"Dispatched job {job_id}")
            except Exception as e:
                logger.error(f"Failed to dispatch job {job_id}: {e}")

    for k8s_job in k8s_jobs.items:
        if not k8s_job.metadata.labels:
            continue
        job_id_str = k8s_job.metadata.labels.get("job-id")
        if not job_id_str:
            continue

        status = k8s_job.status
        if status.succeeded and status.succeeded > 0:
            try:
                batch_v1.delete_namespaced_job(
                    name=k8s_job.metadata.name,
                    namespace=env_config.KUBERNETES_NAMESPACE,
                    propagation_policy="Background",
                )
                logger.info(f"Cleaned up completed k8s Job for job {job_id_str}")
            except Exception as e:
                logger.error(f"Failed to delete completed k8s Job: {e}")

        elif status.failed and status.failed > 0:
            try:
                stats_client.update_job(
                    UUID(job_id_str),
                    JobRequestUpdate(status=JobStatus.failed, error="k8s Job failed after retries"),
                )
                batch_v1.delete_namespaced_job(
                    name=k8s_job.metadata.name,
                    namespace=env_config.KUBERNETES_NAMESPACE,
                    propagation_policy="Background",
                )
                logger.info(f"Marked job {job_id_str} as failed and cleaned up k8s Job")
            except Exception as e:
                logger.error(f"Failed to handle failed k8s Job: {e}")


if __name__ == "__main__":
    init_logging()
    suppress_noisy_logs()
    env_config = get_job_runner_config()
    if not all(
        [env_config.KUBERNETES_NAMESPACE, env_config.DOCKER_IMAGE, env_config.MACHINE_TOKEN, env_config.BACKEND_URL]
    ):
        logger.error("Missing required environment variables")
    else:
        kopf.run()
