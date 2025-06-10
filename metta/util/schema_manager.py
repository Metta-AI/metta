import abc
from datetime import datetime
from typing import List, LiteralString, Sequence

from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine


class Migration(abc.ABC):
    @abc.abstractmethod
    def version(self) -> int:
        pass

    @abc.abstractmethod
    def description(self) -> str:
        pass

    @abc.abstractmethod
    def up(self, conn: Connection) -> None:
        pass


class SqlMigration(Migration):
    def __init__(self, version: int, description: str, sql_statements: List[LiteralString]):
        self._version = version
        self._description = description
        self._sql_statements = sql_statements

    def version(self) -> int:
        return self._version

    def description(self) -> str:
        return self._description

    def up(self, conn: Connection) -> None:
        for stmt in self._sql_statements:
            conn.execute(text(stmt))


class MigrationRecord(BaseModel):
    version: int
    description: str
    applied_at: datetime


def validate_migrations(migrations: Sequence[Migration]):
    for i, migration in enumerate(migrations):
        if migration.version() != i:
            raise ValueError(f"Migration {i} has version {migration.version()}")


migrations_ddl_sqlalchemy = """
CREATE TABLE IF NOT EXISTS migrations (
    version INTEGER PRIMARY KEY,
    description TEXT NOT NULL,
    applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


def get_last_applied_migration(conn: Engine) -> MigrationRecord | None:
    with conn.connect() as sqlalchemy_conn:
        result = sqlalchemy_conn.execute(
            text("SELECT version, description, applied_at FROM migrations ORDER BY version DESC LIMIT 1")
        )
        row = result.fetchone()
        if row:
            return MigrationRecord(version=row[0], description=row[1], applied_at=row[2])
        return None


def init_migrations_table(conn: Engine):
    with conn.begin() as sqlalchemy_conn:
        sqlalchemy_conn.execute(text(migrations_ddl_sqlalchemy))


def run_migrations(conn: Engine, migrations: Sequence[Migration]) -> None:
    validate_migrations(migrations)
    init_migrations_table(conn)
    last_applied_migration = get_last_applied_migration(conn)
    last_applied_migration_version = last_applied_migration.version if last_applied_migration else -1
    for migration in migrations[last_applied_migration_version + 1 :]:
        with conn.begin() as sqlalchemy_conn:
            migration.up(sqlalchemy_conn)
            sqlalchemy_conn.execute(
                text(
                    "INSERT INTO migrations (version, description, applied_at) VALUES (:version, :description, :applied_at)"
                ),
                {
                    "version": migration.version(),
                    "description": migration.description(),
                    "applied_at": datetime.now(),
                },
            )
