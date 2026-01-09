# TODO: These models are partial representations of the episodes/episode_policies/episode_policy_metrics tables.
# Currently only used by tournament code. Migrate other raw SQL queries to use these models.

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import Column, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, Relationship, SQLModel


class Episode(SQLModel, table=True):
    __tablename__ = "episodes"  # type: ignore[assignment]

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    internal_id: int | None = Field(default=None, sa_column_kwargs={"autoincrement": True, "unique": True})
    replay_url: str | None = None
    attributes: dict[str, Any] | None = Field(default=None, sa_column=Column(JSONB))
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), sa_column_kwargs={"server_default": text("now()")}
    )

    episode_policies: list["EpisodePolicy"] = Relationship(back_populates="episode")


class EpisodePolicy(SQLModel, table=True):
    __tablename__ = "episode_policies"  # type: ignore[assignment]

    episode_id: UUID = Field(foreign_key="episodes.id", primary_key=True)
    policy_version_id: UUID = Field(foreign_key="policy_versions.id", primary_key=True)
    num_agents: int

    episode: Episode = Relationship(back_populates="episode_policies")


class EpisodePolicyMetric(SQLModel, table=True):
    __tablename__ = "episode_policy_metrics"  # type: ignore[assignment]

    episode_internal_id: int = Field(foreign_key="episodes.internal_id", primary_key=True)
    pv_internal_id: int = Field(foreign_key="policy_versions.internal_id", primary_key=True)
    metric_name: str = Field(primary_key=True)
    value: float
