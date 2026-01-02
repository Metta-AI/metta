from pydantic_settings import BaseSettings


class RefereeSettings(BaseSettings):
    STATS_SERVER_URI: str = "http://localhost:8000"
    STATS_DB_URI: str = "postgres://postgres:password@127.0.0.1:5432/metta"
    POLL_INTERVAL_SECONDS: float = 30.0
    SELFPLAY_MATCHES: int = 3
    TOP_K: int = 10
    PAIR_MATCHES: int = 3
    MAX_MATCHES_PER_CYCLE: int = 10


settings = RefereeSettings()
