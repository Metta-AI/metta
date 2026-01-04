from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from contextvars import ContextVar
from functools import wraps
from typing import Annotated, ParamSpec, TypeVar

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from metta.app_backend.config import settings

P = ParamSpec("P")
T = TypeVar("T")

_engine = None
_session_factory = None
_current_session: ContextVar[AsyncSession | None] = ContextVar("current_session", default=None)


def _get_async_url(db_uri: str) -> str:
    if db_uri.startswith("postgresql://") or db_uri.startswith("postgres://"):
        return "postgresql+psycopg_async://" + db_uri.split("://", 1)[1]
    return db_uri


def _get_engine():
    global _engine
    if _engine is None:
        async_url = _get_async_url(settings.STATS_DB_URI)
        _engine = create_async_engine(async_url, pool_size=5, max_overflow=10)
    return _engine


def _get_session_factory() -> async_sessionmaker[AsyncSession]:
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(_get_engine(), class_=AsyncSession, expire_on_commit=False)
    return _session_factory


def get_db() -> AsyncSession:
    session = _current_session.get()
    if session is None:
        raise RuntimeError("No database session in context. Use db_session() context manager.")
    return session


@asynccontextmanager
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    existing = _current_session.get()
    if existing is not None:
        yield existing
        return

    factory = _get_session_factory()
    async with factory() as session:
        token = _current_session.set(session)
        try:
            yield session
        finally:
            _current_session.reset(token)


def with_db(func: Callable[P, T]) -> Callable[P, T]:
    @wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        async with db_session():
            return await func(*args, **kwargs)  # type: ignore[misc]

    return wrapper  # type: ignore[return-value]


async def _db_dependency() -> AsyncGenerator[AsyncSession, None]:
    async with db_session() as session:
        yield session


DbSession = Annotated[AsyncSession, Depends(_db_dependency)]
