from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    STATS_DB_URI: str = "postgres://postgres:password@127.0.0.1/postgres"
    DEBUG_USER_EMAIL: str | None = None
    AUTH_SECRET: str | None = Field(default=None, validation_alias="OBSERVATORY_AUTH_SECRET")
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    ANTHROPIC_API_KEY: str | None = None
    LOGIN_SERVICE_URL: str = "https://softmax.com"


settings = Settings()
