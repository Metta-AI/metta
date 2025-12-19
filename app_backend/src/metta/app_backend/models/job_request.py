from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import Column, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, SQLModel


class JobType(str, Enum):
    episode = "episode"


class JobStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


class JobRequest(SQLModel, table=True):
    __tablename__ = "job_requests"  # type: ignore[assignment]

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    job_type: JobType
    job: dict[str, Any] = Field(sa_column=Column(JSONB, nullable=False))
    status: JobStatus = Field(default=JobStatus.pending)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), sa_column_kwargs={"server_default": text("now()")}
    )
    started_at: datetime | None = None
    finished_at: datetime | None = None
    worker: str | None = None
    result: dict[str, Any] | None = Field(default=None, sa_column=Column(JSONB))
    error: str | None = None
