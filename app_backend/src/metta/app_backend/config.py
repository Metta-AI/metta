import os

stats_db_uri = os.getenv("STATS_DB_URI", "postgres://postgres:password@127.0.0.1/postgres")
debug_user_email = os.getenv("DEBUG_USER_EMAIL")

host = os.getenv("HOST", "127.0.0.1")
port = int(os.getenv("PORT", "8000"))
