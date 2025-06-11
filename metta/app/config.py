import os

from metta.app.stats_repo import StatsRepo

stats_db_uri = os.getenv("STATS_DB_URI")
if stats_db_uri is None:
    stats_db_uri = "postgres://postgres:password@127.0.0.1/postgres"

host = os.getenv("HOST")
if host is None:
    host = "0.0.0.0"
port = os.getenv("PORT")
if port is None:
    port = 8000

stats_repo = StatsRepo(stats_db_uri)
