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
    dispatched = "dispatched"
    running = "running"
    completed = "completed"
    failed = "failed"


class _JobRequestBase(SQLModel):
    job_type: JobType
    job: dict[str, Any] = Field(sa_column=Column(JSONB, nullable=False))


class JobRequestCreate(_JobRequestBase):
    pass


class JobRequestUpdate(SQLModel):
    status: JobStatus | None = Field(default=None, description="Tracks k8s-lifecycle status, not semantic job status")
    worker: str | None = Field(default=None, description="Name of the worker that started the job")
    result: dict[str, Any] | None = Field(
        default=None, sa_column=Column(JSONB), description="Contains job-specific results, including possibly errors"
    )
    error: str | None = Field(
        default=None, exclude=True, description="Tracks k8s-lifecycle errors, not semantic job errors"
    )


class JobRequest(_JobRequestBase, JobRequestUpdate, table=True):
    __tablename__ = "job_requests"  # type: ignore[assignment]

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    status: JobStatus = Field(default=JobStatus.pending, nullable=False)
    user_id: str
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), sa_column_kwargs={"server_default": text("now()")}
    )
    dispatched_at: datetime | None = None
    running_at: datetime | None = None
    completed_at: datetime | None = None
