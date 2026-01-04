from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import Column, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.hybrid import hybrid_property
from sqlmodel import Field, SQLModel

# SQLModel + Pydantic multiple-models pattern for FastAPI.
# See: https://sqlmodel.tiangolo.com/tutorial/fastapi/multiple-models/
#
# A convention to stick to for our codebase:
#   - _<Name>Base: shared fields (no table=True). Not to be exposed to other files
#   - <Name>Create: fields needed for creation, extends Base
#   - <Name>Update: fields needed for updates
#   - <Name>: defines columns of the db table (table=True). Extends Base and possibly others, adds other fields
#   - <Name>Public (optional): model for api responses. Note that FastAPI will both:
#         - auto-marshall between <Name> and <Name>Public based on the return type annotation on the endpoint.
#         - exclude Fields with `exclude=True` from the API response.


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
    model_config = {"ignored_types": (hybrid_property,)}  # type: ignore[misc]

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    status: JobStatus = Field(default=JobStatus.pending, nullable=False)
    user_id: str
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), sa_column_kwargs={"server_default": text("now()")}
    )
    dispatched_at: datetime | None = None
    running_at: datetime | None = None
    completed_at: datetime | None = None

    @hybrid_property
    def episode_id(self) -> str | None:  # type: ignore[no-redef]
        if self.result and isinstance(self.result, dict):
            return self.result.get("episode_id")
        return None

    @episode_id.expression  # type: ignore[no-redef]
    def episode_id(cls):
        return cls.result["episode_id"].astext  # type: ignore[union-attr]
