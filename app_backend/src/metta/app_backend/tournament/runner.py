import importlib
import logging
from dataclasses import dataclass
from uuid import UUID

from metta.app_backend.clients.stats_client import StatsClient
from metta.app_backend.metta_repo import MatchPlayerRow, PoolPlayerWithPolicy, PoolRow
from metta.app_backend.tournament.interfaces import (
    MatchPlayer,
    MatchRequest,
    Pool,
    PoolPlayer,
    RefereeInterface,
)
from metta.app_backend.tournament.interfaces import (
    MatchWithEvalStatus as InterfaceMatchWithEvalStatus,
)

logger = logging.getLogger(__name__)


def load_class(class_path: str) -> type:
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def pool_row_to_interface(row: PoolRow) -> Pool:
    return Pool(
        id=row.id,
        season_id=row.season_id,
        name=row.name,
        referee_class=row.referee_class,
        created_at=row.created_at,
        attributes=row.attributes,
    )


def pool_player_to_interface(row: PoolPlayerWithPolicy) -> PoolPlayer:
    return PoolPlayer(
        id=row.id,
        policy_version_id=row.policy_version_id,
        pool_id=row.pool_id,
        added_at=row.added_at,
        removed_at=row.removed_at,
        retired=row.retired,
        attributes=row.attributes,
    )


def match_player_row_to_interface(row: MatchPlayerRow) -> MatchPlayer:
    return MatchPlayer(
        match_id=row.match_id,
        policy_version_id=row.policy_version_id,
        position=row.position,
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
    def __init__(self, client: StatsClient):
        self.client = client
        self._referee_cache: dict[str, RefereeInterface] = {}

    def get_referee(self, class_path: str, attributes: dict | None = None) -> RefereeInterface:
        cache_key = class_path
        if cache_key not in self._referee_cache:
            cls = load_class(class_path)
            self._referee_cache[cache_key] = cls(**(attributes or {}))
        return self._referee_cache[cache_key]

    def run_referee_for_pool(self, pool_id: UUID) -> PoolRunResult:
        errors: list[str] = []
        matches_created = 0

        pool_response = self.client.get_pool(pool_id)
        pool_row = pool_response.pool

        try:
            referee = self.get_referee(pool_row.referee_class, pool_row.attributes)
        except Exception as e:
            logger.exception(f"Failed to load referee class {pool_row.referee_class}")
            return PoolRunResult(pool_id=pool_id, matches_created=0, errors=[f"Failed to load referee: {e}"])

        pool = pool_row_to_interface(pool_row)

        players_response = self.client.get_pool_players(pool_id, include_removed=False)
        pool_players = [pool_player_to_interface(p) for p in players_response.players]

        matches_response = self.client.get_matches_for_pool_with_eval_status(pool_id, limit=10000)
        match_history = [
            InterfaceMatchWithEvalStatus(
                id=m.id,
                pool_id=m.pool_id,
                eval_task_id=m.eval_task_id,
                created_at=m.created_at,
                players=[match_player_row_to_interface(p) for p in m.players],
                eval_status=m.eval_status,
                eval_is_finished=m.eval_is_finished,
            )
            for m in matches_response.matches
        ]

        try:
            desired_matches: list[MatchRequest] = referee.get_desired_matches(pool, pool_players, match_history)
        except Exception as e:
            logger.exception(f"Referee.get_desired_matches failed for pool {pool_id}")
            errors.append(f"Referee error: {e}")
            return PoolRunResult(pool_id=pool_id, matches_created=0, errors=errors)

        for match_request in desired_matches:
            try:
                self.client.create_match(
                    pool_id=pool_id,
                    policy_version_ids=match_request.policy_version_ids,
                )
                matches_created += 1
            except Exception as e:
                logger.exception(f"Failed to create match for pool {pool_id}")
                errors.append(f"Match creation error: {e}")

        return PoolRunResult(pool_id=pool_id, matches_created=matches_created, errors=errors)

    def run_all_referees(self) -> TournamentRunResult:
        all_errors: list[str] = []
        pool_results: list[PoolRunResult] = []
        total_matches = 0

        seasons_response = self.client.get_seasons(limit=1000)
        pool_ids: set[UUID] = set()

        for season in seasons_response.seasons:
            pools_response = self.client.get_pools_for_season(season.id)
            for pool in pools_response.pools:
                pool_ids.add(pool.id)

        for pool_id in pool_ids:
            result = self.run_referee_for_pool(pool_id)
            pool_results.append(result)
            total_matches += result.matches_created
            all_errors.extend(result.errors)

        return TournamentRunResult(
            pools_processed=len(pool_ids),
            total_matches_created=total_matches,
            pool_results=pool_results,
            errors=all_errors,
        )
