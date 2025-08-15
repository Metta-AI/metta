from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatadogConfig(BaseSettings):
    model_config = SettingsConfigDict(
        extra="ignore",
    )

    # Core settings
    DD_TRACE_ENABLED: bool = Field(default=False, description="Enable Datadog tracing")
    DD_SERVICE: str = Field(default="metta", description="Service name")
    DD_ENV: str = Field(default="development", description="Environment (production, staging, development)")
    DD_VERSION: str | None = Field(default=None, description="Service version (e.g., git commit hash)")

    # Agent connection
    DD_AGENT_HOST: str | None = Field(default=None, description="Datadog agent hostname")
    DD_TRACE_AGENT_PORT: int = Field(default=8126, description="Datadog trace agent port")
    DD_TRACE_AGENT_URL: str | None = Field(default=None, description="Full URL of trace agent (overrides host/port)")

    # Additional settings
    DD_SITE: str = Field(default="datadoghq.com", description="Datadog site")

    def to_env_dict(self) -> dict[str, str]:
        return self.model_dump(mode="json", exclude_none=True)


datadog_config = DatadogConfig()
