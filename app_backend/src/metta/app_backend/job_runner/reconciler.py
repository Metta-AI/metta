import asyncio
import logging
from uuid import UUID

from kubernetes import client, config

from metta.app_backend.clients.stats_client import StatsClient
from metta.app_backend.job_runner.config import (
    JOB_NAMESPACE,
    LABEL_APP,
    LABEL_APP_VALUE,
    LABEL_JOB_ID,
    get_dispatch_config,
)
from metta.app_backend.models.job_request import JobRequestUpdate, JobStatus
from metta.common.auth.auth_config_reader_writer import observatory_auth_config
from metta.common.util.log_config import init_logging, suppress_noisy_logs

logger = logging.getLogger(__name__)


def get_k8s_client() -> client.BatchV1Api:
    try:
        config.load_incluster_config()
    except config.ConfigException:
        config.load_kube_config()
    return client.BatchV1Api()


async def run_reconciler(poll_interval: float = 10.0):
    cfg = get_dispatch_config()
    batch_v1 = get_k8s_client()

    observatory_auth_config.save_token(cfg.MACHINE_TOKEN, cfg.BACKEND_URL)
    stats_client = StatsClient(cfg.BACKEND_URL)

    logger.info(f"Reconciler started: backend_url={cfg.BACKEND_URL}, namespace={JOB_NAMESPACE}")

    try:
        while True:
            try:
                await reconcile_once(batch_v1, stats_client)
            except Exception as e:
                logger.error(f"Reconcile error: {e}", exc_info=True)
            await asyncio.sleep(poll_interval)
    finally:
        stats_client.close()


async def reconcile_once(batch_v1: client.BatchV1Api, stats_client: StatsClient):
    k8s_jobs = batch_v1.list_namespaced_job(
        namespace=JOB_NAMESPACE,
        label_selector=f"{LABEL_APP}={LABEL_APP_VALUE}",
    )

    for k8s_job in k8s_jobs.items:
        job_id_str = k8s_job.metadata.labels.get(LABEL_JOB_ID)
        if not job_id_str:
            continue

        job_id = UUID(job_id_str)
        status = k8s_job.status

        if status.succeeded and status.succeeded > 0:
            try:
                batch_v1.delete_namespaced_job(
                    name=k8s_job.metadata.name,
                    namespace=JOB_NAMESPACE,
                    propagation_policy="Background",
                )
                logger.info(f"Cleaned up succeeded job {job_id}")
            except Exception as e:
                logger.error(f"Failed to delete job {job_id}: {e}")

        elif status.failed and status.failed >= (k8s_job.spec.backoff_limit or 0):
            try:
                stats_client.update_job(
                    job_id,
                    JobRequestUpdate(status=JobStatus.failed, error="k8s Job failed"),
                )
                batch_v1.delete_namespaced_job(
                    name=k8s_job.metadata.name,
                    namespace=JOB_NAMESPACE,
                    propagation_policy="Background",
                )
                logger.info(f"Marked failed and cleaned up job {job_id}")
            except Exception as e:
                logger.error(f"Failed to handle failed job {job_id}: {e}")


if __name__ == "__main__":
    init_logging()
    suppress_noisy_logs()
    asyncio.run(run_reconciler())
