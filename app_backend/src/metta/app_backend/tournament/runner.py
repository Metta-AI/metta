import importlib
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID

from metta.app_backend.metta_repo import (
    MatchWithEvalStatus,
    MettaRepo,
    PoolPlayerRow,
    PoolRow,
    SeasonRow,
)
from metta.app_backend.tournament.interfaces import (
    CommissionerInterface,
    MatchPlayer,
    MatchRequest,
    PolicyVersion,
    Pool,
    PoolPlayer,
    PoolPlayerMutation,
    RefereeInterface,
    Season,
    SeasonScorerInterface,
)
from metta.app_backend.tournament.interfaces import (
    MatchWithEvalStatus as InterfaceMatchWithEvalStatus,
)

logger = logging.getLogger(__name__)


def load_class(class_path: str) -> type:
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def season_row_to_interface(row: SeasonRow) -> Season:
    return Season(
        id=row.id,
        name=row.name,
        commissioner_class=row.commissioner_class,
        scorer_class=row.scorer_class,
        created_at=row.created_at,
        attributes=row.attributes,
    )


def pool_row_to_interface(row: PoolRow) -> Pool:
    return Pool(
        id=row.id,
        season_id=row.season_id,
        name=row.name,
        referee_class=row.referee_class,
        created_at=row.created_at,
        attributes=row.attributes,
    )


def pool_player_row_to_interface(row: PoolPlayerRow) -> PoolPlayer:
    return PoolPlayer(
        id=row.id,
        policy_version_id=row.policy_version_id,
        pool_id=row.pool_id,
        added_at=row.added_at,
        removed_at=row.removed_at,
        retired=row.retired,
        attributes=row.attributes,
    )


def match_row_to_interface(row: MatchWithEvalStatus) -> InterfaceMatchWithEvalStatus:
    return InterfaceMatchWithEvalStatus(
        id=row.id,
        pool_id=row.pool_id,
        eval_task_id=row.eval_task_id,
        created_at=row.created_at,
        players=[
            MatchPlayer(
                match_id=p.match_id,
                policy_version_id=p.policy_version_id,
                position=p.position,
            )
            for p in row.players
        ],
        eval_status=row.eval_status,
        eval_is_finished=row.eval_is_finished,
    )


@dataclass
class PoolRunResult:
    pool_id: UUID
    matches_created: int
    errors: list[str]


@dataclass
class TournamentRunResult:
    pools_processed: int
    total_matches_created: int
    pool_results: list[PoolRunResult]
    errors: list[str]


class TournamentRunner:
    def __init__(self, repo: MettaRepo):
        self.repo = repo
        self._commissioner_cache: dict[str, CommissionerInterface] = {}
        self._referee_cache: dict[str, RefereeInterface] = {}
        self._scorer_cache: dict[str, SeasonScorerInterface] = {}

    def get_commissioner(self, class_path: str, attributes: dict[str, Any] | None = None) -> CommissionerInterface:
        cache_key = class_path
        if cache_key not in self._commissioner_cache:
            cls = load_class(class_path)
            self._commissioner_cache[cache_key] = cls(**(attributes or {}))
        return self._commissioner_cache[cache_key]

    def get_referee(self, class_path: str, attributes: dict[str, Any] | None = None) -> RefereeInterface:
        cache_key = class_path
        if cache_key not in self._referee_cache:
            cls = load_class(class_path)
            self._referee_cache[cache_key] = cls(**(attributes or {}))
        return self._referee_cache[cache_key]

    def get_scorer(self, class_path: str, attributes: dict[str, Any] | None = None) -> SeasonScorerInterface:
        cache_key = class_path
        if cache_key not in self._scorer_cache:
            cls = load_class(class_path)
            self._scorer_cache[cache_key] = cls(**(attributes or {}))
        return self._scorer_cache[cache_key]

    async def run_referee_for_pool(self, pool_id: UUID) -> PoolRunResult:
        errors: list[str] = []
        matches_created = 0

        pool_row = await self.repo.get_pool(pool_id)
        if pool_row is None:
            return PoolRunResult(pool_id=pool_id, matches_created=0, errors=["Pool not found"])

        try:
            referee = self.get_referee(pool_row.referee_class, pool_row.attributes)
        except Exception as e:
            logger.exception(f"Failed to load referee class {pool_row.referee_class}")
            return PoolRunResult(pool_id=pool_id, matches_created=0, errors=[f"Failed to load referee: {e}"])

        pool = pool_row_to_interface(pool_row)

        player_rows = await self.repo.get_pool_players(pool_id, include_removed=False)
        pool_players = [
            PoolPlayer(
                id=p.id,
                policy_version_id=p.policy_version_id,
                pool_id=p.pool_id,
                added_at=p.added_at,
                removed_at=p.removed_at,
                retired=p.retired,
                attributes=p.attributes,
            )
            for p in player_rows
        ]

        match_rows = await self.repo.get_matches_for_pool_with_eval_status(pool_id, limit=10000)
        match_history = [match_row_to_interface(m) for m in match_rows]

        try:
            desired_matches: list[MatchRequest] = referee.get_desired_matches(pool, pool_players, match_history)
        except Exception as e:
            logger.exception(f"Referee.get_desired_matches failed for pool {pool_id}")
            errors.append(f"Referee error: {e}")
            return PoolRunResult(pool_id=pool_id, matches_created=0, errors=errors)

        for match_request in desired_matches:
            try:
                await self.repo.create_match(
                    pool_id=pool_id,
                    policy_version_ids=match_request.policy_version_ids,
                )
                matches_created += 1
            except Exception as e:
                logger.exception(f"Failed to create match for pool {pool_id}")
                errors.append(f"Match creation error: {e}")

        return PoolRunResult(pool_id=pool_id, matches_created=matches_created, errors=errors)

    async def run_all_referees(self) -> TournamentRunResult:
        all_errors: list[str] = []
        pool_results: list[PoolRunResult] = []
        total_matches = 0

        seasons = await self.repo.get_seasons(limit=1000)
        pool_ids: set[UUID] = set()

        for season in seasons:
            pools = await self.repo.get_pools_for_season(season.id)
            for pool in pools:
                pool_ids.add(pool.id)

        for pool_id in pool_ids:
            result = await self.run_referee_for_pool(pool_id)
            pool_results.append(result)
            total_matches += result.matches_created
            all_errors.extend(result.errors)

        return TournamentRunResult(
            pools_processed=len(pool_ids),
            total_matches_created=total_matches,
            pool_results=pool_results,
            errors=all_errors,
        )

    async def validate_and_submit_to_season(
        self,
        policy_version_id: UUID,
        season_id: UUID,
        user_id: str,
    ) -> tuple[UUID | None, str | None]:
        season_row = await self.repo.get_season(season_id)
        if season_row is None:
            return None, "Season not found"

        season = season_row_to_interface(season_row)

        pv_row = await self.repo.get_policy_version_with_name(policy_version_id)
        if pv_row is None:
            return None, "Policy version not found"

        policy_version = PolicyVersion(
            id=pv_row.id,
            policy_id=pv_row.policy_id,
            version=pv_row.version,
            user_id=user_id,
            name=pv_row.name,
            created_at=pv_row.created_at,
            attributes=pv_row.attributes,
        )

        try:
            commissioner = self.get_commissioner(season_row.commissioner_class, season_row.attributes)
        except Exception as e:
            logger.exception(f"Failed to load commissioner class {season_row.commissioner_class}")
            return None, f"Failed to load commissioner: {e}"

        allowed, rejection_reason = commissioner.validate_submission(
            policy_version=policy_version,
            season=season,
            submitter_user_id=user_id,
            now=datetime.utcnow(),
        )

        if not allowed:
            return None, rejection_reason or "Submission rejected"

        pools = await self.repo.get_pools_for_season(season_id)
        if not pools:
            return None, "Season has no pools configured"

        pool_interfaces = [pool_row_to_interface(p) for p in pools]

        try:
            target_pool = commissioner.get_target_pool(policy_version, season, pool_interfaces)
        except Exception as e:
            logger.exception(f"Commissioner.get_target_pool failed for season {season_id}")
            return None, f"Failed to determine target pool: {e}"

        player_id = await self.repo.add_pool_player(
            policy_version_id=policy_version_id,
            pool_id=target_pool.id,
        )

        existing_players = await self.repo.get_pool_players(target_pool.id)
        existing_policies: list[tuple[PolicyVersion, float]] = []
        for p in existing_players:
            if p.policy_user_id == user_id and p.policy_version_id != policy_version_id:
                pv = PolicyVersion(
                    id=p.policy_version_id,
                    policy_id=UUID(int=0),
                    version=p.version,
                    user_id=p.policy_user_id,
                    name=p.policy_name,
                    created_at=p.added_at,
                    attributes=p.attributes,
                )
                existing_policies.append((pv, 0.0))

        try:
            mutations = commissioner.on_new_submission(
                policy_version=policy_version,
                pool=target_pool,
                existing_policies=existing_policies,
            )
            await self._apply_mutations(mutations)
        except Exception:
            logger.exception(f"Commissioner.on_new_submission failed for policy {policy_version_id}")

        return player_id, None

    async def _apply_mutations(self, mutations: list[PoolPlayerMutation]) -> None:
        for mutation in mutations:
            if mutation.action == "retire" and mutation.target_pool_id:
                await self.repo.retire_pool_player(
                    policy_version_id=mutation.policy_version_id,
                    pool_id=mutation.target_pool_id,
                )
            elif mutation.action == "remove" and mutation.target_pool_id:
                await self.repo.remove_pool_player(
                    policy_version_id=mutation.policy_version_id,
                    pool_id=mutation.target_pool_id,
                )
            elif mutation.action == "add" and mutation.target_pool_id:
                await self.repo.add_pool_player(
                    policy_version_id=mutation.policy_version_id,
                    pool_id=mutation.target_pool_id,
                )
