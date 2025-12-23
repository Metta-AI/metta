from datetime import datetime

from metta.app_backend.tournament.interfaces import (
    CommissionerInterface,
    PolicyVersion,
    Pool,
    PoolPlayer,
    PoolPlayerMutation,
    Season,
)


class DefaultCommissioner(CommissionerInterface):
    def validate_submission(
        self,
        policy_version: PolicyVersion,
        season: Season,
        submitter_user_id: str,
        now: datetime,
    ) -> tuple[bool, str | None]:
        return True, None

    def get_target_pool(
        self,
        policy_version: PolicyVersion,
        season: Season,
        pools: list[Pool],
    ) -> Pool:
        if not pools:
            raise ValueError(f"Season {season.name} has no pools")
        return pools[0]

    def on_pool_closed(
        self,
        pool: Pool,
        season: Season,
        pool_players: list[PoolPlayer],
        rankings: list[tuple[PolicyVersion, float]],
    ) -> list[PoolPlayerMutation]:
        return []

    def on_new_submission(
        self,
        policy_version: PolicyVersion,
        pool: Pool,
        existing_policies: list[tuple[PolicyVersion, float]],
    ) -> list[PoolPlayerMutation]:
        return []
