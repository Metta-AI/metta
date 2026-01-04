from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Optional
from uuid import UUID, uuid4

from sqlalchemy import Column, text
from sqlalchemy.dialects.postgresql import ARRAY, INTEGER
from sqlmodel import Field, Relationship, SQLModel

if TYPE_CHECKING:
    from metta.app_backend.models.job_request import JobRequest


class MatchStatus(str, Enum):
    pending = "pending"
    scheduled = "scheduled"
    running = "running"
    completed = "completed"
    failed = "failed"


class MembershipAction(str, Enum):
    add = "add"
    retire = "retire"


class _SeasonBase(SQLModel):
    name: str = Field(index=True, unique=True)
    description: str | None = None


class SeasonCreate(_SeasonBase):
    pass


class Season(_SeasonBase, table=True):
    __tablename__ = "seasons"  # type: ignore[assignment]

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), sa_column_kwargs={"server_default": text("now()")}
    )

    pools: list["Pool"] = Relationship(back_populates="season")


class _PoolBase(SQLModel):
    name: str | None = None


class PoolCreate(_PoolBase):
    season_id: UUID | None = None


class Pool(_PoolBase, table=True):
    __tablename__ = "pools"  # type: ignore[assignment]

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    season_id: UUID | None = Field(foreign_key="seasons.id", nullable=True, index=True)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), sa_column_kwargs={"server_default": text("now()")}
    )

    season: Season | None = Relationship(back_populates="pools")
    matches: list["Match"] = Relationship(back_populates="pool")


class _PoolPlayerBase(SQLModel):
    pool_id: UUID = Field(foreign_key="pools.id", index=True)
    # FK exists in DB (migrations.py) but can't declare here - no SQLModel class for policy_versions
    policy_version_id: UUID = Field(index=True)


class PoolPlayerCreate(_PoolPlayerBase):
    pass


class PoolPlayer(_PoolPlayerBase, table=True):
    __tablename__ = "pool_players"  # type: ignore[assignment]

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
    __tablename__ = "matches"  # type: ignore[assignment]

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    job_id: UUID | None = Field(foreign_key="job_requests.id", nullable=True, index=True)
    status: MatchStatus = Field(default=MatchStatus.pending)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), sa_column_kwargs={"server_default": text("now()")}
    )
    completed_at: datetime | None = None

    pool: Pool = Relationship(back_populates="matches")
    players: list["MatchPlayer"] = Relationship(back_populates="match", sa_relationship_kwargs={"lazy": "selectin"})
    job: Optional["JobRequest"] = Relationship()


class _MatchPlayerBase(SQLModel):
    match_id: UUID = Field(foreign_key="matches.id", index=True)
    # FK exists in DB (migrations.py) but can't declare here - no SQLModel class for policy_versions
    policy_version_id: UUID = Field(index=True)
    policy_index: int = Field(default=0)


class MatchPlayerCreate(_MatchPlayerBase):
    pass


class MatchPlayerUpdate(SQLModel):
    score: float | None = None


class MatchPlayer(_MatchPlayerBase, table=True):
    __tablename__ = "match_players"  # type: ignore[assignment]

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    score: float | None = None

    match: Match = Relationship(back_populates="players")


class MembershipChangeRecord(SQLModel, table=True):
    __tablename__ = "membership_changes"  # type: ignore[assignment]

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    pool_id: UUID = Field(foreign_key="pools.id", index=True)
    # FK exists in DB (migrations.py) but can't declare here - no SQLModel class for policy_versions
    policy_version_id: UUID = Field(index=True)
    action: MembershipAction
    notes: str | None = None
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), sa_column_kwargs={"server_default": text("now()")}
    )
