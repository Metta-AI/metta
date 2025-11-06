import abc
import datetime
import typing

import psycopg
import pydantic


class Migration(abc.ABC):
    @abc.abstractmethod
    def version(self) -> int:
        pass

    @abc.abstractmethod
    def description(self) -> str:
        pass

    @abc.abstractmethod
    def up(self, conn: psycopg.Connection) -> None:
        pass

    @abc.abstractmethod
    async def up_async(self, conn: psycopg.AsyncConnection) -> None:
        pass


class SqlMigration(Migration):
    def __init__(self, version: int, description: str, sql_statements: typing.List[typing.LiteralString]):
        self._version = version
        self._description = description
        self._sql_statements = sql_statements

    def version(self) -> int:
        return self._version

    def description(self) -> str:
        return self._description

    def up(self, conn: psycopg.Connection) -> None:
        with conn.cursor() as cursor:
            for stmt in self._sql_statements:
                cursor.execute(psycopg.sql.SQL(stmt))

    async def up_async(self, conn: psycopg.AsyncConnection) -> None:
        async with conn.cursor() as cursor:
            for stmt in self._sql_statements:
                await cursor.execute(psycopg.sql.SQL(stmt))


class MigrationRecord(pydantic.BaseModel):
    version: int
    description: str
    applied_at: datetime.datetime


def validate_migrations(migrations: typing.Sequence[Migration]):
    for i, migration in enumerate(migrations):
        if migration.version() != i:
            raise ValueError(f"Migration {i} has version {migration.version()}")


migrations_ddl = psycopg.sql.SQL("""
CREATE TABLE IF NOT EXISTS migrations (
    version INTEGER PRIMARY KEY,
    description TEXT NOT NULL,
    applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
""")

last_applied_migration_query = psycopg.sql.SQL(
    "SELECT version, description, applied_at FROM migrations ORDER BY version DESC LIMIT 1"
)

insert_migration_query = psycopg.sql.SQL(
    "INSERT INTO migrations (version, description, applied_at) VALUES (%s, %s, %s)"
)


def get_last_applied_migration(conn: psycopg.Connection) -> MigrationRecord | None:
    with conn.cursor() as cursor:
        cursor.execute(last_applied_migration_query)
        result = cursor.fetchone()
        if result:
            [version, description, applied_at] = result
            return MigrationRecord(version=version, description=description, applied_at=applied_at)
        return None


async def get_last_applied_migration_async(conn: psycopg.AsyncConnection) -> MigrationRecord | None:
    async with conn.cursor() as cursor:
        await cursor.execute(last_applied_migration_query)
        result = await cursor.fetchone()
        if result:
            [version, description, applied_at] = result
            return MigrationRecord(version=version, description=description, applied_at=applied_at)
        return None


def init_migrations_table(conn: psycopg.Connection):
    with conn.cursor() as cursor:
        cursor.execute(migrations_ddl)
        conn.commit()


async def init_migrations_table_async(conn: psycopg.AsyncConnection):
    async with conn.cursor() as cursor:
        await cursor.execute(migrations_ddl)
        await conn.commit()


def run_migrations(conn: psycopg.Connection, migrations: typing.Sequence[Migration]) -> None:
    validate_migrations(migrations)
    init_migrations_table(conn)
    last_applied_migration = get_last_applied_migration(conn)
    last_applied_migration_version = last_applied_migration.version if last_applied_migration else -1
    for migration in migrations[last_applied_migration_version + 1 :]:
        with conn.transaction():
            migration.up(conn)
            with conn.cursor() as cursor:
                cursor.execute(
                    insert_migration_query, (migration.version(), migration.description(), datetime.datetime.now())
                )

    conn.commit()


async def run_migrations_async(conn: psycopg.AsyncConnection, migrations: typing.Sequence[Migration]) -> None:
    validate_migrations(migrations)
    await init_migrations_table_async(conn)
    last_applied_migration = await get_last_applied_migration_async(conn)
    last_applied_migration_version = last_applied_migration.version if last_applied_migration else -1
    for migration in migrations[last_applied_migration_version + 1 :]:
        async with conn.transaction():
            await migration.up_async(conn)
            async with conn.cursor() as cursor:
                await cursor.execute(
                    insert_migration_query, (migration.version(), migration.description(), datetime.datetime.now())
                )

    await conn.commit()
