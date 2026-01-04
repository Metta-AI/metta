# TODO: These models are partial representations of the policies/policy_versions tables.
# Currently only used by tournament code. Migrate other raw SQL queries to use these models.

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import Column, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, Relationship, SQLModel


class Policy(SQLModel, table=True):
    __tablename__ = "policies"  # type: ignore[assignment]

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    name: str
    user_id: str
    attributes: dict[str, Any] | None = Field(default=None, sa_column=Column(JSONB))
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), sa_column_kwargs={"server_default": text("now()")}
    )

    versions: list["PolicyVersion"] = Relationship(back_populates="policy")


class PolicyVersion(SQLModel, table=True):
    __tablename__ = "policy_versions"  # type: ignore[assignment]

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    internal_id: int | None = Field(default=None, sa_column_kwargs={"autoincrement": True, "unique": True})
    policy_id: UUID = Field(foreign_key="policies.id", index=True)
    version: int
    s3_path: str | None = None
    attributes: dict[str, Any] | None = Field(default=None, sa_column=Column(JSONB))
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), sa_column_kwargs={"server_default": text("now()")}
    )

    policy: Policy = Relationship(back_populates="versions")
