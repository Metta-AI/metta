from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from metta.app_backend.config import settings

_engine = None
_session_factory = None


def _get_async_url(db_uri: str) -> str:
    if db_uri.startswith("postgresql://") or db_uri.startswith("postgres://"):
        return "postgresql+psycopg_async://" + db_uri.split("://", 1)[1]
    return db_uri


def get_engine():
    global _engine
    if _engine is None:
        async_url = _get_async_url(settings.STATS_DB_URI)
        _engine = create_async_engine(async_url, pool_size=5, max_overflow=10)
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(get_engine(), class_=AsyncSession, expire_on_commit=False)
    return _session_factory


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    factory = get_session_factory()
    async with factory() as session:
        yield session
