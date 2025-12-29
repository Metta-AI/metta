import functools
import logging

from kubernetes import client
from kubernetes import config as kubernetes_config

from metta.app_backend.job_runner.config import (
    JOB_NAMESPACE,
    LABEL_APP,
    LABEL_APP_VALUE,
    LABEL_JOB_ID,
    get_dispatch_config,
)
from metta.app_backend.models.job_request import JobRequest, JobType

logger = logging.getLogger(__name__)


@functools.cache
def get_k8s_client() -> client.BatchV1Api:
    cfg = get_dispatch_config()

    if cfg.LOCAL_DEV:
        if not cfg.LOCAL_DEV_K8S_CONTEXT:
            raise ValueError("LOCAL_DEV=true requires LOCAL_DEV_K8S_CONTEXT to be set")
        kubernetes_config.load_kube_config(context=cfg.LOCAL_DEV_K8S_CONTEXT)
    else:
        # Prod: require in-cluster config, no silent fallback
        kubernetes_config.load_incluster_config()

    return client.BatchV1Api()


def dispatch_job(job: JobRequest) -> str:
    if job.job_type == JobType.episode:
        return create_episode_job(job)
    raise ValueError(f"Unknown job type: {job.job_type}")


def create_episode_job(job: JobRequest) -> str:
    cfg = get_dispatch_config()
    batch_v1 = get_k8s_client()
    job_name = f"job-{job.id.hex[:8]}"

    labels = {
        LABEL_APP: LABEL_APP_VALUE,
        LABEL_JOB_ID: str(job.id),
        # TODO(Nishad): Create EKS Fargate Profile with selector (namespace=jobs, compute=fargate)
        # "compute": "fargate",
    }

    volumes: list[client.V1Volume] = []
    volume_mounts: list[client.V1VolumeMount] = []

    if cfg.LOCAL_DEV and cfg.LOCAL_DEV_MOUNTS:
        for i, mount in enumerate(cfg.LOCAL_DEV_MOUNTS.split(",")):
            parts = mount.strip().split(":")
            if len(parts) != 2:
                logger.warning(f"Invalid mount format: {mount}, expected 'host:container'")
                continue
            host_path, container_path = parts
            vol_name = f"local-mount-{i}"
            volumes.append(
                client.V1Volume(
                    name=vol_name,
                    host_path=client.V1HostPathVolumeSource(path=host_path),
                )
            )
            volume_mounts.append(client.V1VolumeMount(name=vol_name, mount_path=container_path))

    k8s_job = client.V1Job(
        metadata=client.V1ObjectMeta(
            name=job_name,
            namespace=JOB_NAMESPACE,
            labels=labels,
        ),
        spec=client.V1JobSpec(
            # No retries for now
            backoff_limit=0,
            # Kill job if it runs longer than 1 hour
            active_deadline_seconds=3600,
            # Auto-delete job 1 hour after completion (backup; watcher deletes immediately)
            # Longer TTL gives watcher time to catch up if it restarts
            ttl_seconds_after_finished=3600,
            template=client.V1PodTemplateSpec(
                metadata=client.V1ObjectMeta(labels=labels),
                spec=client.V1PodSpec(
                    restart_policy="Never",
                    volumes=volumes if volumes else None,
                    containers=[
                        client.V1Container(
                            name="worker",
                            image=cfg.EPISODE_RUNNER_IMAGE,
                            image_pull_policy="IfNotPresent",
                            command=[
                                "uv",
                                "run",
                                "--no-sync",
                                "python",
                                "-m",
                                "metta.sim.single_episode_runner",
                                str(job.id),
                            ],
                            env=[
                                client.V1EnvVar(name="STATS_SERVER_URI", value=cfg.STATS_SERVER_URI),
                                client.V1EnvVar(name="MACHINE_TOKEN", value=cfg.MACHINE_TOKEN),
                            ]
                            + (
                                [client.V1EnvVar(name="AWS_PROFILE", value=cfg.LOCAL_DEV_AWS_PROFILE)]
                                if cfg.LOCAL_DEV and cfg.LOCAL_DEV_AWS_PROFILE
                                else []
                            ),
                            volume_mounts=volume_mounts if volume_mounts else None,
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

    batch_v1.create_namespaced_job(namespace=JOB_NAMESPACE, body=k8s_job)
    logger.info(f"Created k8s Job {job_name} for job {job.id}")
    return job_name
