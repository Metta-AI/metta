import logging
import os

import urllib3
from kubernetes import client, config

from metta.app_backend.job_runner.config import (
    JOB_NAMESPACE,
    LABEL_APP,
    LABEL_APP_VALUE,
    LABEL_JOB_ID,
    get_dispatch_config,
)
from metta.app_backend.models.job_request import JobRequest, JobType

logger = logging.getLogger(__name__)


def get_k8s_client() -> client.BatchV1Api:
    try:
        config.load_incluster_config()
    except config.ConfigException:
        config.load_kube_config()

    # For local dev via Docker, disable SSL verification (cert is for 127.0.0.1, not host.docker.internal)
    if os.environ.get("KUBERNETES_SKIP_TLS_VERIFY", "").lower() == "true":
        configuration = client.Configuration.get_default_copy()
        configuration.verify_ssl = False
        client.Configuration.set_default(configuration)
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

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

    k8s_job = client.V1Job(
        metadata=client.V1ObjectMeta(
            name=job_name,
            namespace=JOB_NAMESPACE,
            labels=labels,
        ),
        spec=client.V1JobSpec(
            backoff_limit=3,
            active_deadline_seconds=3600,
            ttl_seconds_after_finished=300,
            template=client.V1PodTemplateSpec(
                metadata=client.V1ObjectMeta(labels=labels),
                spec=client.V1PodSpec(
                    restart_policy="Never",
                    containers=[
                        client.V1Container(
                            name="worker",
                            image=cfg.EPISODE_RUNNER_IMAGE,
                            image_pull_policy="IfNotPresent",
                            command=[
                                "/workspace/metta/.venv/bin/python",
                                "-m",
                                "metta.app_backend.job_runner.episode_job_worker",
                                str(job.id),
                            ],
                            env=[
                                client.V1EnvVar(name="BACKEND_URL", value=cfg.BACKEND_URL),
                                client.V1EnvVar(name="MACHINE_TOKEN", value=cfg.MACHINE_TOKEN),
                                client.V1EnvVar(name="METTA_SCHEME_SERVER_URI", value=cfg.METTA_SCHEME_SERVER_URI),
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

    batch_v1.create_namespaced_job(namespace=JOB_NAMESPACE, body=k8s_job)
    logger.info(f"Created k8s Job {job_name} for job {job.id}")
    return job_name
