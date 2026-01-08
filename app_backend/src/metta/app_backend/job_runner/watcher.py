import functools
import logging
import time
from typing import Literal, TypedDict
from uuid import UUID

from kubernetes import (
    client,
    watch,  # type: ignore[attr-defined]
)
from kubernetes import config as kubernetes_config
from kubernetes.client.rest import ApiException
from opentelemetry import trace as otel_trace

from metta.app_backend.clients.stats_client import StatsClient
from metta.app_backend.health_server import start_health_server, update_heartbeat
from metta.app_backend.job_runner.config import (
    JOB_NAMESPACE,
    LABEL_APP,
    LABEL_APP_VALUE,
    LABEL_JOB_ID,
    get_dispatch_config,
)
from metta.app_backend.models.job_request import JobRequestUpdate, JobStatus
from metta.common.otel.tracing import init_otel_tracing, trace
from metta.common.util.log_config import init_logging, suppress_noisy_logs

logger = logging.getLogger(__name__)

WATCH_TIMEOUT_SECONDS = 30


@functools.cache
def _get_k8s_clients() -> tuple[client.CoreV1Api, client.BatchV1Api]:
    cfg = get_dispatch_config()
    if cfg.LOCAL_DEV:
        if not cfg.LOCAL_DEV_K8S_CONTEXT:
            raise ValueError("LOCAL_DEV=true requires LOCAL_DEV_K8S_CONTEXT to be set")
        kubernetes_config.load_kube_config(context=cfg.LOCAL_DEV_K8S_CONTEXT)
    else:
        kubernetes_config.load_incluster_config()
    return client.CoreV1Api(), client.BatchV1Api()


# ADDED: Pod created (usually starts in Pending phase)
# MODIFIED: Pod state changed (phase transitions, container status updates)
# DELETED: Pod removed from cluster
# BOOKMARK: Internal watch checkpoint (no actual change, just resourceVersion update)
# ERROR: Watch stream error
K8sPodWatchEventType = Literal["ADDED", "MODIFIED", "DELETED", "BOOKMARK", "ERROR"]


class K8sPodWatchEvent(TypedDict):
    type: K8sPodWatchEventType
    object: client.V1Pod


def run_watcher():
    cfg = get_dispatch_config()
    _get_k8s_clients()

    start_health_server()

    stats_client = StatsClient(backend_url=cfg.STATS_SERVER_URI, machine_token=cfg.MACHINE_TOKEN)
    stats_client._validate_authenticated()
    logger.info(f"Watcher started: stats_server_uri={cfg.STATS_SERVER_URI}, namespace={JOB_NAMESPACE}")

    try:
        while True:
            try:
                _watch_pods(stats_client)
            except Exception as e:
                logger.error(f"Watch error, restarting: {e}", exc_info=True)
                time.sleep(1)
    finally:
        stats_client.close()


def _watch_pods(stats_client: StatsClient):
    label_selector = f"{LABEL_APP}={LABEL_APP_VALUE}"
    core_v1, _ = _get_k8s_clients()

    pod_list = core_v1.list_namespaced_pod(namespace=JOB_NAMESPACE, label_selector=label_selector)
    if not pod_list.metadata or not pod_list.metadata.resource_version:
        logger.error(f"Invalid pod list: {pod_list}")
        return

    for pod in pod_list.items:
        _handle_pod_state(stats_client, pod)

    resource_version = pod_list.metadata.resource_version
    logger.info(f"Starting pod watch from resourceVersion={resource_version}")
    update_heartbeat()

    w = watch.Watch()
    event: K8sPodWatchEvent
    for event in w.stream(  # type: ignore[assignment]
        core_v1.list_namespaced_pod,
        namespace=JOB_NAMESPACE,
        label_selector=label_selector,
        resource_version=resource_version,
        timeout_seconds=WATCH_TIMEOUT_SECONDS,
    ):
        update_heartbeat()
        event_type, pod = event["type"], event["object"]
        if event_type in ("ADDED", "MODIFIED"):
            _handle_pod_state(stats_client, pod)
        elif event_type == "DELETED":
            _handle_pod_deleted(stats_client, pod)


def _get_job_info(pod: client.V1Pod) -> tuple[UUID, str] | None:
    if not pod.metadata or not pod.metadata.labels:
        return None
    job_id_str = pod.metadata.labels.get(LABEL_JOB_ID)
    if not job_id_str:
        return None
    return UUID(job_id_str), pod.metadata.name or "unknown"


@trace("tournament.job.status_update")
def _handle_pod_state(stats_client: StatsClient, pod: client.V1Pod):
    info = _get_job_info(pod)
    if not info or not pod.status:
        return

    job_id, pod_name = info
    phase = pod.status.phase

    span = otel_trace.get_current_span()
    if span.is_recording():
        span.set_attribute("job.id", str(job_id))
        span.set_attribute("pod.name", pod_name)
        span.set_attribute("pod.phase", phase)

    if phase == "Succeeded":
        _update_job_status(stats_client, job_id, JobStatus.completed)
        _delete_k8s_job_for_pod(pod)
        logger.info(f"Job {job_id} completed (pod {pod_name})")
    elif phase == "Failed":
        error = _get_pod_error(pod)
        _update_job_status(stats_client, job_id, JobStatus.failed, error=error)
        _delete_k8s_job_for_pod(pod)
        logger.info(f"Job {job_id} failed (pod {pod_name}): {error}")
    elif phase == "Running" and _is_container_running(pod):
        _update_job_status(stats_client, job_id, JobStatus.running, worker=pod_name)
        logger.debug(f"Job {job_id} running (pod {pod_name})")


def _handle_pod_deleted(stats_client: StatsClient, pod: client.V1Pod):
    info = _get_job_info(pod)
    if not info:
        return

    phase = pod.status.phase if pod.status else None
    if phase in ("Succeeded", "Failed"):
        return

    job_id, pod_name = info
    _update_job_status(stats_client, job_id, JobStatus.failed, error="Pod deleted unexpectedly")
    logger.warning(f"Job {job_id} failed: pod {pod_name} deleted unexpectedly (phase={phase})")


def _is_container_running(pod: client.V1Pod) -> bool:
    if not pod.status or not pod.status.container_statuses:
        return False
    return any(cs.state and cs.state.running for cs in pod.status.container_statuses)


def _get_pod_error(pod: client.V1Pod) -> str:
    if pod.status and pod.status.container_statuses:
        for cs in pod.status.container_statuses:
            if cs.state and cs.state.terminated and cs.state.terminated.reason:
                return cs.state.terminated.reason
    return (pod.status.message if pod.status else None) or "Pod failed"


def _get_job_name_for_pod(pod: client.V1Pod) -> str | None:
    if not pod.metadata or not pod.metadata.owner_references:
        return None
    return next((ref.name for ref in pod.metadata.owner_references if ref.kind == "Job"), None)


def _delete_k8s_job_for_pod(pod: client.V1Pod):
    job_name = _get_job_name_for_pod(pod)
    if not job_name:
        return
    try:
        _, batch_v1 = _get_k8s_clients()
        batch_v1.delete_namespaced_job(name=job_name, namespace=JOB_NAMESPACE, propagation_policy="Background")
    except ApiException as e:
        if e.status == 404:
            logger.debug(f"K8s job {job_name} already deleted")
        else:
            logger.error(f"Failed to delete k8s job {job_name}: {e}")
    except Exception as e:
        logger.error(f"Failed to delete k8s job {job_name}: {e}")


def _update_job_status(
    stats_client: StatsClient,
    job_id: UUID,
    status: JobStatus,
    error: str | None = None,
    worker: str | None = None,
):
    try:
        current = stats_client.get_job(job_id)
        if current.status == status:
            return
        if current.status in (JobStatus.completed, JobStatus.failed):
            # Job already in terminal state, but still update error if we have one (e.g., OOMKilled)
            if error and not current.error:
                stats_client.update_job(job_id, JobRequestUpdate(error=error))
            return
        stats_client.update_job(job_id, JobRequestUpdate(status=status, error=error, worker=worker))
    except Exception as e:
        logger.error(f"Failed to update job {job_id} status to {status}: {e}")


if __name__ == "__main__":
    init_logging()
    suppress_noisy_logs()
    init_otel_tracing(service_name="job-watcher")
    run_watcher()
