from abc import ABC, abstractmethod
from uuid import UUID

from pydantic import BaseModel
from sqlmodel import select

from metta.app_backend.models.tournament import Match, MatchPlayer, MatchStatus, PoolPlayer
from mettagrid.config.mettagrid_config import MettaGridConfig


class MatchData(BaseModel):
    match_id: UUID
    status: MatchStatus
    pool_player_ids: list[UUID]
    assignments: list[int] = []


class MatchRequest(BaseModel):
    pool_player_ids: list[UUID]
    assignments: list[int]
    env: MettaGridConfig
    episode_tags: dict[str, str] = {}


class ScoredMatchData(BaseModel):
    match_id: UUID
    policy_scores: dict[UUID, float]
    assignments: list[int]
    policy_version_ids: list[UUID]
    episode_tags: dict[str, str] = {}


class ScorerInterface(ABC):
    @abstractmethod
    def compute_scores(
        self,
        policy_version_ids: list[UUID],
        matches: list[ScoredMatchData],
    ) -> dict[UUID, float]:
        pass


class RefereeBase(ABC):
    scorer: ScorerInterface

    @abstractmethod
    def get_matches_to_schedule(
        self,
        players: list[PoolPlayer],
        matches: list[MatchData],
    ) -> list[MatchRequest]:
        pass

    async def get_leaderboard(self, pool_id: UUID) -> list[tuple[UUID, float, int]]:
        """Returns list of (policy_version_id, score, match_count) sorted by score descending."""
        from sqlalchemy.orm import selectinload

        from metta.app_backend.database import get_db

        session = get_db()
        matches = list(
            (
                await session.execute(
                    select(Match)
                    .where(Match.pool_id == pool_id)
                    .where(Match.status == MatchStatus.completed)
                    .options(selectinload(Match.players).selectinload(MatchPlayer.pool_player))  # type: ignore[arg-type]
                )
            )
            .scalars()
            .all()
        )

        if not matches:
            return []

        all_policy_ids: set[UUID] = set()
        scored_matches: list[ScoredMatchData] = []
        match_counts: dict[UUID, int] = {}

        for match in matches:
            if not match.players or any(mp.score is None for mp in match.players):
                continue

            policy_scores: dict[UUID, float] = {}
            policy_version_ids: list[UUID] = []
            for mp in sorted(match.players, key=lambda x: x.policy_index):
                pv_id = mp.pool_player.policy_version_id
                policy_scores[pv_id] = mp.score  # type: ignore[assignment]
                if mp.policy_index >= len(policy_version_ids):
                    policy_version_ids.append(pv_id)
                all_policy_ids.add(pv_id)
                match_counts[pv_id] = match_counts.get(pv_id, 0) + 1

            scored_matches.append(
                ScoredMatchData(
                    match_id=match.id,
                    policy_scores=policy_scores,
                    assignments=match.assignments,
                    policy_version_ids=policy_version_ids,
                )
            )

        if not scored_matches:
            return []

        scores = self.scorer.compute_scores(list(all_policy_ids), scored_matches)
        results = [(pv, score, match_counts.get(pv, 0)) for pv, score in scores.items()]
        return sorted(results, key=lambda x: x[1], reverse=True)
