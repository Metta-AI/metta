from pydantic_settings import BaseSettings

POLL_INTERVAL_SECONDS: float = 30.0
POLL_INTERVAL_FAST_SECONDS: float = 2.0
MAX_MATCHES_PER_CYCLE: int = 10
PROMOTION_MIN_SCORE: float = 1


class CommissionerSettings(BaseSettings):
    STATS_SERVER_URI: str = "http://localhost:8000"
    STATS_DB_URI: str = "postgres://postgres:password@127.0.0.1:5432/metta"


settings = CommissionerSettings()
