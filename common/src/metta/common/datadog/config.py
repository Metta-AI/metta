import pydantic
import pydantic_settings


class DatadogConfig(pydantic_settings.BaseSettings):
    model_config = pydantic_settings.SettingsConfigDict(
        extra="ignore",
    )

    # Core settings
    DD_TRACE_ENABLED: bool = pydantic.Field(default=False, description="Enable Datadog tracing")
    DD_SERVICE: str = pydantic.Field(default="metta", description="Service name")
    DD_ENV: str = pydantic.Field(default="development", description="Environment (production, staging, development)")
    DD_VERSION: str | None = pydantic.Field(default=None, description="Service version (e.g., git commit hash)")

    # Agent connection
    DD_AGENT_HOST: str | None = pydantic.Field(default=None, description="Datadog agent hostname")
    DD_TRACE_AGENT_PORT: int = pydantic.Field(default=8126, description="Datadog trace agent port")
    DD_TRACE_AGENT_URL: str | None = pydantic.Field(
        default=None, description="Full URL of trace agent (overrides host/port)"
    )

    # Additional settings
    DD_SITE: str = pydantic.Field(default="datadoghq.com", description="Datadog site")

    def to_env_dict(self) -> dict[str, str]:
        return self.model_dump(mode="json", exclude_none=True)


datadog_config = DatadogConfig()
