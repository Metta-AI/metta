from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Optional
from uuid import UUID, uuid4

from sqlalchemy import Column, text
from sqlalchemy.dialects.postgresql import ARRAY, INTEGER
from sqlmodel import Field, Relationship, SQLModel

from metta.app_backend.models.policies import PolicyVersion

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
    remove = "remove"


class Season(SQLModel, table=True):
    __tablename__ = "seasons"  # type: ignore[assignment]

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    name: str = Field(index=True, unique=True)
    description: str | None = None
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), sa_column_kwargs={"server_default": text("now()")}
    )

    pools: list["Pool"] = Relationship(back_populates="season")


class Pool(SQLModel, table=True):
    __tablename__ = "pools"  # type: ignore[assignment]

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    season_id: UUID | None = Field(foreign_key="seasons.id", nullable=True, index=True)
    name: str | None = None
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), sa_column_kwargs={"server_default": text("now()")}
    )

    season: Season | None = Relationship(back_populates="pools")
    matches: list["Match"] = Relationship(back_populates="pool")
    players: list["PoolPlayer"] = Relationship(
        back_populates="pool",
        sa_relationship_kwargs={"lazy": "selectin"},
    )

    @property
    def active_players(self) -> list["PoolPlayer"]:
        return [p for p in self.players if not p.retired]

    @property
    def active_member_ids(self) -> set[UUID]:
        return {p.policy_version_id for p in self.players if not p.retired}


class PoolPlayer(SQLModel, table=True):
    __tablename__ = "pool_players"  # type: ignore[assignment]

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    pool_id: UUID = Field(foreign_key="pools.id", index=True)
    policy_version_id: UUID = Field(foreign_key="policy_versions.id", index=True)
    retired: bool = Field(default=False)

    pool: "Pool" = Relationship(back_populates="players")
    policy_version: PolicyVersion = Relationship(back_populates="pool_players")
    membership_changes: list["MembershipChange"] = Relationship(back_populates="pool_player")


class Match(SQLModel, table=True):
    __tablename__ = "matches"  # type: ignore[assignment]

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    pool_id: UUID = Field(foreign_key="pools.id", index=True)
    job_id: UUID | None = Field(foreign_key="job_requests.id", nullable=True, index=True)
    assignments: list[int] = Field(sa_column=Column(ARRAY(INTEGER), nullable=False))
    status: MatchStatus = Field(default=MatchStatus.pending)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), sa_column_kwargs={"server_default": text("now()")}
    )
    completed_at: datetime | None = None

    pool: Pool = Relationship(back_populates="matches")
    players: list["MatchPlayer"] = Relationship(back_populates="match", sa_relationship_kwargs={"lazy": "selectin"})
    job: Optional["JobRequest"] = Relationship()


class MatchPlayer(SQLModel, table=True):
    __tablename__ = "match_players"  # type: ignore[assignment]

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    match_id: UUID = Field(foreign_key="matches.id", index=True)
    pool_player_id: UUID = Field(foreign_key="pool_players.id", index=True)
    policy_index: int = Field(default=0)
    score: float | None = None

    match: Match = Relationship(back_populates="players")
    pool_player: PoolPlayer = Relationship()


class MembershipChange(SQLModel, table=True):
    __tablename__ = "membership_changes"  # type: ignore[assignment]

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    pool_player_id: UUID = Field(foreign_key="pool_players.id", index=True)
    action: MembershipAction
    notes: str | None = None
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), sa_column_kwargs={"server_default": text("now()")}
    )

    pool_player: PoolPlayer = Relationship(back_populates="membership_changes")
