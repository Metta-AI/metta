from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID


@dataclass
class PolicyVersion:
    id: UUID
    policy_id: UUID
    version: int
    user_id: str
    name: str
    created_at: datetime
    attributes: dict[str, Any]


@dataclass
class Season:
    id: UUID
    name: str
    commissioner_class: str
    scorer_class: str
    created_at: datetime
    attributes: dict[str, Any]


@dataclass
class Pool:
    id: UUID
    season_id: UUID | None
    name: str | None
    referee_class: str
    created_at: datetime
    attributes: dict[str, Any]


@dataclass
class PoolPlayer:
    id: UUID
    policy_version_id: UUID
    pool_id: UUID
    added_at: datetime
    removed_at: datetime | None
    retired: bool
    attributes: dict[str, Any]


@dataclass
class Match:
    id: UUID
    pool_id: UUID
    eval_task_id: int | None
    created_at: datetime


@dataclass
class MatchPlayer:
    match_id: UUID
    policy_version_id: UUID
    position: int


@dataclass
class MatchRequest:
    policy_version_ids: list[UUID]


@dataclass
class PoolPlayerMutation:
    policy_version_id: UUID
    action: str  # 'add', 'remove', 'retire', 'promote'
    target_pool_id: UUID | None = None
    reason: str | None = None


class CommissionerInterface(ABC):
    @abstractmethod
    def validate_submission(
        self,
        policy_version: PolicyVersion,
        season: Season,
        submitter_user_id: str,
        now: datetime,
    ) -> tuple[bool, str | None]:
        pass

    @abstractmethod
    def get_target_pool(
        self,
        policy_version: PolicyVersion,
        season: Season,
        pools: list[Pool],
    ) -> Pool:
        pass

    @abstractmethod
    def on_pool_closed(
        self,
        pool: Pool,
        season: Season,
        pool_players: list[PoolPlayer],
        rankings: list[tuple[PolicyVersion, float]],
    ) -> list[PoolPlayerMutation]:
        pass

    @abstractmethod
    def on_new_submission(
        self,
        policy_version: PolicyVersion,
        pool: Pool,
        existing_policies: list[tuple[PolicyVersion, float]],
    ) -> list[PoolPlayerMutation]:
        pass


class SeasonScorerInterface(ABC):
    @abstractmethod
    def get_pool_rankings(
        self,
        pool: Pool,
        pool_players: list[PoolPlayer],
        match_history: list[Match],
        match_players: dict[UUID, list[MatchPlayer]],
    ) -> list[tuple[PolicyVersion, float]]:
        pass

    @abstractmethod
    def get_season_standings(
        self,
        season: Season,
        pools: list[Pool],
        all_pool_players: list[PoolPlayer],
        all_matches: list[Match],
        all_match_players: dict[UUID, list[MatchPlayer]],
    ) -> list[tuple[PolicyVersion, float]]:
        pass


class RefereeInterface(ABC):
    @abstractmethod
    def get_desired_matches(
        self,
        pool: Pool,
        pool_players: list[PoolPlayer],
        match_history: list[Match],
        match_players: dict[UUID, list[MatchPlayer]],
    ) -> list[MatchRequest]:
        pass
