import abc
from datetime import datetime
from typing import List, LiteralString, Sequence

from psycopg import AsyncConnection, Connection, sql
from pydantic import BaseModel


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

    @abc.abstractmethod
    async def up_async(self, conn: AsyncConnection) -> None:
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
        with conn.cursor() as cursor:
            for stmt in self._sql_statements:
                cursor.execute(sql.SQL(stmt))

    async def up_async(self, conn: AsyncConnection) -> None:
        async with conn.cursor() as cursor:
            for stmt in self._sql_statements:
                await cursor.execute(sql.SQL(stmt))


class MigrationRecord(BaseModel):
    version: int
    description: str
    applied_at: datetime


def validate_migrations(migrations: Sequence[Migration]):
    for i, migration in enumerate(migrations):
        if migration.version() != i:
            raise ValueError(f"Migration {i} has version {migration.version()}")


migrations_ddl = sql.SQL("""
CREATE TABLE IF NOT EXISTS migrations (
    version INTEGER PRIMARY KEY,
    description TEXT NOT NULL,
    applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
""")

last_applied_migration_query = sql.SQL(
    "SELECT version, description, applied_at FROM migrations ORDER BY version DESC LIMIT 1"
)

insert_migration_query = sql.SQL("INSERT INTO migrations (version, description, applied_at) VALUES (%s, %s, %s)")


def get_last_applied_migration(conn: Connection) -> MigrationRecord | None:
    with conn.cursor() as cursor:
        cursor.execute(last_applied_migration_query)
        result = cursor.fetchone()
        if result:
            [version, description, applied_at] = result
            return MigrationRecord(version=version, description=description, applied_at=applied_at)
        return None


async def get_last_applied_migration_async(conn: AsyncConnection) -> MigrationRecord | None:
    async with conn.cursor() as cursor:
        await cursor.execute(last_applied_migration_query)
        result = await cursor.fetchone()
        if result:
            [version, description, applied_at] = result
            return MigrationRecord(version=version, description=description, applied_at=applied_at)
        return None


def init_migrations_table(conn: Connection):
    with conn.cursor() as cursor:
        cursor.execute(migrations_ddl)
        conn.commit()


async def init_migrations_table_async(conn: AsyncConnection):
    async with conn.cursor() as cursor:
        await cursor.execute(migrations_ddl)
        await conn.commit()


def run_migrations(conn: Connection, migrations: Sequence[Migration]) -> None:
    validate_migrations(migrations)
    init_migrations_table(conn)
    last_applied_migration = get_last_applied_migration(conn)
    last_applied_migration_version = last_applied_migration.version if last_applied_migration else -1
    for migration in migrations[last_applied_migration_version + 1 :]:
        with conn.transaction():
            migration.up(conn)
            with conn.cursor() as cursor:
                cursor.execute(insert_migration_query, (migration.version(), migration.description(), datetime.now()))

    conn.commit()


async def run_migrations_async(conn: AsyncConnection, migrations: Sequence[Migration]) -> None:
    validate_migrations(migrations)
    await init_migrations_table_async(conn)
    last_applied_migration = await get_last_applied_migration_async(conn)
    last_applied_migration_version = last_applied_migration.version if last_applied_migration else -1
    for migration in migrations[last_applied_migration_version + 1 :]:
        async with conn.transaction():
            await migration.up_async(conn)
            async with conn.cursor() as cursor:
                await cursor.execute(
                    insert_migration_query, (migration.version(), migration.description(), datetime.now())
                )

    await conn.commit()
