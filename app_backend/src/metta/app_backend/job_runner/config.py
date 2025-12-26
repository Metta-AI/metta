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
    # Local dev mode: comma-separated host:container mount pairs
    # e.g. "~/.aws:/root/.aws,path/to/repo:/workspace/metta"
    # AWS_PROFILE is also passed to job pods when LOCAL_DEV_MOUNTS is set
    LOCAL_DEV_MOUNTS: str = ""
    LOCAL_DEV_AWS_PROFILE: str = ""


@lru_cache
def get_dispatch_config() -> JobDispatchConfig:
    return JobDispatchConfig()
