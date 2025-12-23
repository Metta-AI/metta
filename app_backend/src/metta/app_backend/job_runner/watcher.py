import logging
from uuid import UUID

from kubernetes import client, watch

from metta.app_backend.clients.stats_client import StatsClient
from metta.app_backend.job_runner.config import (
    JOB_NAMESPACE,
    LABEL_APP,
    LABEL_APP_VALUE,
    LABEL_JOB_ID,
    get_dispatch_config,
)
from metta.app_backend.job_runner.dispatcher import get_k8s_client
from metta.app_backend.models.job_request import JobRequestUpdate, JobStatus
from metta.common.auth.auth_config_reader_writer import observatory_auth_config
from metta.common.util.log_config import init_logging, suppress_noisy_logs

logger = logging.getLogger(__name__)


def run_watcher():
    cfg = get_dispatch_config()
    batch_v1 = get_k8s_client()

    observatory_auth_config.save_token(cfg.MACHINE_TOKEN, cfg.BACKEND_URL)
    stats_client = StatsClient(cfg.BACKEND_URL)

    logger.info(f"Watcher started: backend_url={cfg.BACKEND_URL}, namespace={JOB_NAMESPACE}")

    try:
        while True:
            try:
                watch_jobs(batch_v1, stats_client)
            except Exception as e:
                logger.error(f"Watch error, restarting: {e}", exc_info=True)
    finally:
        stats_client.close()


def watch_jobs(batch_v1: client.BatchV1Api, stats_client: StatsClient):
    label_selector = f"{LABEL_APP}={LABEL_APP_VALUE}"

    job_list = batch_v1.list_namespaced_job(namespace=JOB_NAMESPACE, label_selector=label_selector)
    resource_version = job_list.metadata.resource_version
    logger.info(f"Starting watch from resourceVersion={resource_version}")

    w = watch.Watch()
    for event in w.stream(
        batch_v1.list_namespaced_job,
        namespace=JOB_NAMESPACE,
        label_selector=label_selector,
        resource_version=resource_version,
    ):
        event_type = event["type"]
        k8s_job = event["object"]

        job_id_str = k8s_job.metadata.labels.get(LABEL_JOB_ID)
        if not job_id_str:
            continue

        job_id = UUID(job_id_str)
        job_name = k8s_job.metadata.name

        logger.debug(f"Event {event_type} for job {job_name} (id={job_id})")

        if event_type in ("ADDED", "MODIFIED"):
            handle_job_state(batch_v1, stats_client, job_id, k8s_job)
        elif event_type == "DELETED":
            logger.info(f"Job {job_name} deleted")


def handle_job_state(
    batch_v1: client.BatchV1Api,
    stats_client: StatsClient,
    job_id: UUID,
    k8s_job: client.V1Job,
):
    status = k8s_job.status
    job_name = k8s_job.metadata.name
    backoff_limit = k8s_job.spec.backoff_limit or 0

    if status.succeeded and status.succeeded > 0:
        update_job_status(stats_client, job_id, JobStatus.completed)
        delete_k8s_job(batch_v1, job_name)
        logger.info(f"Job {job_id} completed")

    elif status.failed and status.failed >= backoff_limit:
        update_job_status(stats_client, job_id, JobStatus.failed, error="k8s job failed")
        delete_k8s_job(batch_v1, job_name)
        logger.info(f"Job {job_id} failed")

    elif status.active and status.active > 0:
        update_job_status(stats_client, job_id, JobStatus.running)
        logger.debug(f"Job {job_id} running")


def update_job_status(stats_client: StatsClient, job_id: UUID, status: JobStatus, error: str | None = None):
    try:
        current = stats_client.get_job(job_id)
        if current.status == status:
            return
        if is_terminal(current.status):
            return

        stats_client.update_job(job_id, JobRequestUpdate(status=status, error=error))
    except Exception as e:
        logger.error(f"Failed to update job {job_id} status to {status}: {e}")


def is_terminal(status: JobStatus) -> bool:
    return status in (JobStatus.completed, JobStatus.failed)


def delete_k8s_job(batch_v1: client.BatchV1Api, job_name: str):
    try:
        batch_v1.delete_namespaced_job(
            name=job_name,
            namespace=JOB_NAMESPACE,
            propagation_policy="Background",
        )
    except Exception as e:
        logger.error(f"Failed to delete k8s job {job_name}: {e}")


if __name__ == "__main__":
    init_logging()
    suppress_noisy_logs()
    run_watcher()
