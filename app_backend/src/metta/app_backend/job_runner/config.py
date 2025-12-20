import os

from pydantic_settings import BaseSettings


class JobRunnerConfig(BaseSettings):
    BACKEND_URL: str
    MACHINE_TOKEN: str
    DOCKER_IMAGE: str
    KUBERNETES_NAMESPACE: str
    HOSTNAME: str = "unknown"


def get_job_runner_config() -> JobRunnerConfig:
    return JobRunnerConfig.model_validate(os.environ)
