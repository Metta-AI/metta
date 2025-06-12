import os


def env_var(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None:
        return default
    return value


stats_db_uri = env_var("STATS_DB_URI", "postgres://postgres:password@127.0.0.1/postgres")

host = env_var("HOST", "0.0.0.0")
port = int(env_var("PORT", "8000"))
