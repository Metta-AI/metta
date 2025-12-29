from functools import lru_cache

from pydantic_settings import BaseSettings

JOB_NAMESPACE = "jobs"

LABEL_APP = "app"
LABEL_APP_VALUE = "episode-runner"
LABEL_JOB_ID = "job-id"


class JobDispatchConfig(BaseSettings):
    EPISODE_RUNNER_IMAGE: str = ""
    STATS_SERVER_URI: str = ""
    # TODO: limit the scope of this to only update the job in question
    MACHINE_TOKEN: str = ""

    # Local dev mode: enables kube_config loading and volume mounts
    # When False (prod), requires in-cluster config - no silent fallback
    LOCAL_DEV: bool = False
    # Required when LOCAL_DEV=True: validates we're using the right k8s context
    LOCAL_DEV_K8S_CONTEXT: str = ""
    # Optional volume mounts for hot-reloading code in job pods
    # Format: comma-separated host:container pairs, e.g. "~/.aws:/root/.aws"
    LOCAL_DEV_MOUNTS: str = ""
    LOCAL_DEV_AWS_PROFILE: str = ""


@lru_cache
def get_dispatch_config() -> JobDispatchConfig:
    return JobDispatchConfig()
