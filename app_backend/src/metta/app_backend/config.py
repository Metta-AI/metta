from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    STATS_DB_URI: str = "postgres://postgres:password@127.0.0.1/postgres"
    OBSERVATORY_AUTH_SECRET: str | None = None
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    ANTHROPIC_API_KEY: str | None = None
    LOGIN_SERVICE_URL: str = "https://softmax.com"
    RUN_MIGRATIONS: bool = Field(default=False, description="Run migrations on startup")


settings = Settings()
