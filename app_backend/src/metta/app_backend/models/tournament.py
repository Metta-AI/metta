from datetime import UTC, datetime
from enum import Enum
from uuid import UUID, uuid4

from sqlalchemy import Column, text
from sqlalchemy.dialects.postgresql import ARRAY, INTEGER
from sqlmodel import Field, SQLModel


class MatchStatus(str, Enum):
    pending = "pending"
    scheduled = "scheduled"
    running = "running"
    completed = "completed"
    failed = "failed"


class _SeasonBase(SQLModel):
    name: str = Field(index=True)
    description: str | None = None


class SeasonCreate(_SeasonBase):
    pass


class Season(_SeasonBase, table=True):
    __tablename__ = "seasons"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), sa_column_kwargs={"server_default": text("now()")}
    )


class _PoolBase(SQLModel):
    name: str | None = None
    is_academy: bool = Field(default=False)


class PoolCreate(_PoolBase):
    season_id: UUID | None = None


class Pool(_PoolBase, table=True):
    __tablename__ = "pools"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    season_id: UUID | None = Field(foreign_key="seasons.id", nullable=True, index=True)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), sa_column_kwargs={"server_default": text("now()")}
    )


class _PoolPlayerBase(SQLModel):
    pool_id: UUID = Field(foreign_key="pools.id", index=True)
    policy_version_id: UUID = Field(foreign_key="policy_versions.id", index=True)


class PoolPlayerCreate(_PoolPlayerBase):
    pass


class PoolPlayer(_PoolPlayerBase, table=True):
    __tablename__ = "pool_players"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    added_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), sa_column_kwargs={"server_default": text("now()")}
    )
    removed_at: datetime | None = None
    retired: bool = Field(default=False)


class _MatchBase(SQLModel):
    pool_id: UUID = Field(foreign_key="pools.id", index=True)
    environment_name: str = Field(default="machina1_open_world")
    assignments: list[int] = Field(sa_column=Column(ARRAY(INTEGER), nullable=False))


class MatchCreate(_MatchBase):
    pass


class MatchUpdate(SQLModel):
    job_id: UUID | None = None
    status: MatchStatus | None = None
    completed_at: datetime | None = None


class Match(_MatchBase, table=True):
    __tablename__ = "matches"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    job_id: UUID | None = Field(foreign_key="job_requests.id", nullable=True, index=True)
    status: MatchStatus = Field(default=MatchStatus.pending)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), sa_column_kwargs={"server_default": text("now()")}
    )
    completed_at: datetime | None = None


class _MatchPlayerBase(SQLModel):
    match_id: UUID = Field(foreign_key="matches.id", index=True)
    policy_version_id: UUID = Field(foreign_key="policy_versions.id", index=True)
    policy_index: int = Field(default=0)


class MatchPlayerCreate(_MatchPlayerBase):
    pass


class MatchPlayerUpdate(SQLModel):
    score: float | None = None


class MatchPlayer(_MatchPlayerBase, table=True):
    __tablename__ = "match_players"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    score: float | None = None
