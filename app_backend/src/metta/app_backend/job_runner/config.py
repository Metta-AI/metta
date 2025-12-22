from functools import lru_cache

from pydantic_settings import BaseSettings

JOB_NAMESPACE = "jobs"

LABEL_APP = "app"
LABEL_APP_VALUE = "episode-runner"
LABEL_JOB_ID = "job-id"


class JobDispatchConfig(BaseSettings):
    EPISODE_RUNNER_IMAGE: str = ""
    BACKEND_URL: str = ""
    # TODO(Nishad): Instead of passing MACHINE_TOKEN in, auto-create it via a SystemTokens table
    # that has (name, machine_token_foreign_key), unique on name. Gets-or-creates on startup.
    MACHINE_TOKEN: str = ""


@lru_cache
def get_dispatch_config() -> JobDispatchConfig:
    return JobDispatchConfig()
