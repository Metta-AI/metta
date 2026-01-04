import asyncio
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import UTC, datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.orm import selectinload
from sqlmodel import col, select

from metta.app_backend.clients.stats_client import StatsClient
from metta.app_backend.database import get_db, with_db
from metta.app_backend.models.episodes import Episode, EpisodePolicy, EpisodePolicyMetric
from metta.app_backend.models.job_request import JobRequestCreate, JobStatus, JobType
from metta.app_backend.models.policies import PolicyVersion
from metta.app_backend.models.tournament import (
    Match,
    MatchPlayer,
    MatchStatus,
    MembershipAction,
    MembershipChange,
    Pool,
    PoolPlayer,
    Season,
)
from metta.app_backend.tournament.referees.base import MatchData, MatchRequest, RefereeBase
from metta.app_backend.tournament.settings import (
    MAX_OUTSTANDING_MATCHES,
    POLL_INTERVAL_FAST_SECONDS,
    POLL_INTERVAL_SECONDS,
    settings,
)
from metta.sim.single_episode_runner import SingleEpisodeJob

logger = logging.getLogger(__name__)

SOFTMAX_S3_REPLAYS_PREFIX = "s3://softmax-public/replays/tournament"


class MembershipChangeRequest(BaseModel):
    pool_name: str
    policy_version_id: UUID
    action: Literal["add", "remove"]
    notes: str | None = None


class CommissionerBase(ABC):
    season_name: str
    referees: dict[str, RefereeBase]
    leaderboard_pool: str

    @abstractmethod
    def get_new_submission_membership_changes(self, policy_version_id: UUID) -> list[MembershipChangeRequest]:
        pass

    @abstractmethod
    async def get_membership_changes(self, pools: dict[str, Pool]) -> list[MembershipChangeRequest]:
        pass

    async def run(self) -> None:
        await self._ensure_season_exists()
        logger.info(f"Starting commissioner for season '{self.season_name}' ({self.season_id})")
        while True:
            had_activity = False
            try:
                had_activity = await self._run_cycle()
            except Exception as e:
                logger.error(f"Commissioner cycle error: {e}", exc_info=True)
            interval = POLL_INTERVAL_FAST_SECONDS if had_activity else POLL_INTERVAL_SECONDS
            await asyncio.sleep(interval)

    @with_db
    async def _ensure_season_exists(self) -> None:
        session = get_db()
        season = (await session.execute(select(Season).filter_by(name=self.season_name))).scalar_one_or_none()
        if season:
            self.season_id = season.id
        else:
            season = Season(name=self.season_name)
            session.add(season)
            await session.commit()
            await session.refresh(season)
            self.season_id = season.id
            logger.info(f"Created season '{self.season_name}' with id {self.season_id}")

    @with_db
    async def _run_cycle(self) -> bool:
        """Run one cycle of the commissioner. Returns True if there was activity."""
        pools = await self._ensure_pools_exist(list(self.referees.keys()))

        status_changed = await self._sync_match_statuses()

        all_matches = await self._get_season_matches()
        matches_by_pool: dict[UUID, list[MatchData]] = {pool.id: [] for pool in pools.values()}
        outstanding = 0
        for m in all_matches:
            if m.pool_id in matches_by_pool:
                matches_by_pool[m.pool_id].append(m)
            if m.status in (MatchStatus.pending, MatchStatus.scheduled, MatchStatus.running):
                outstanding += 1

        slots_available = max(0, MAX_OUTSTANDING_MATCHES - outstanding)

        total_scheduled = 0
        for pool_name, pool in pools.items():
            if slots_available <= 0:
                break
            referee = self.referees[pool_name]
            players = await self._get_pool_players(pool.id)
            matches = matches_by_pool[pool.id]

            requests = referee.get_matches_to_schedule(players, matches)
            for req in requests[:slots_available]:
                await self._create_and_dispatch_match(pool.id, req)
                total_scheduled += 1
                slots_available -= 1

        if total_scheduled > 0:
            logger.info(f"Scheduled {total_scheduled} new matches")

        changes = await self.get_membership_changes(pools)
        await self._apply_membership_changes(changes, pools)

        return status_changed or total_scheduled > 0 or len(changes) > 0

    async def _ensure_pools_exist(self, pool_names: list[str]) -> dict[str, Pool]:
        session = get_db()

        existing = list((await session.execute(select(Pool).filter_by(season_id=self.season_id))).scalars().all())
        existing_names = {p.name for p in existing if p.name}

        for name in pool_names:
            if name not in existing_names:
                session.add(Pool(season_id=self.season_id, name=name))
                logger.info(f"Created pool '{name}' for season {self.season_id}")

        await session.commit()

        all_pools = (
            (
                await session.execute(
                    select(Pool).filter_by(season_id=self.season_id).options(selectinload(Pool.players))  # type: ignore[arg-type]
                )
            )
            .scalars()
            .all()
        )
        return {p.name: p for p in all_pools if p.name}

    async def _sync_match_statuses(self) -> bool:
        """Sync match statuses from job statuses. Returns True if any changed."""
        session = get_db()
        pending = list(
            (
                await session.execute(
                    select(Match)
                    .join(Pool, Match.pool_id == Pool.id)  # type: ignore[arg-type]
                    .where(Pool.season_id == self.season_id)
                    .where(col(Match.status).in_([MatchStatus.scheduled, MatchStatus.running]))
                    .options(selectinload(Match.job))  # type: ignore[arg-type]
                )
            )
            .scalars()
            .all()
        )

        updated = 0
        for match in pending:
            job = match.job
            if not job:
                continue
            if job.status == JobStatus.running and match.status != MatchStatus.running:
                match.status = MatchStatus.running
                updated += 1
            elif job.status == JobStatus.completed:
                match.status = MatchStatus.completed
                match.completed_at = datetime.now(UTC)
                updated += 1
            elif job.status == JobStatus.failed:
                match.status = MatchStatus.failed
                updated += 1

        if updated > 0:
            logger.info(f"Updated {updated} match statuses")
            await session.commit()

        scores_updated = await self._sync_match_scores()
        return updated > 0 or scores_updated

    async def _sync_match_scores(self) -> bool:
        """Sync match scores from episode metrics. Returns True if any updated."""
        session = get_db()

        # Find completed matches with missing scores and their episode IDs
        matches_query = text("""
            SELECT DISTINCT m.id AS match_id, jr.result->>'episode_id' AS episode_id
            FROM matches m
            JOIN pools p ON m.pool_id = p.id
            JOIN match_players mp ON mp.match_id = m.id
            JOIN job_requests jr ON jr.id = m.job_id
            WHERE p.season_id = :season_id
                AND m.status = 'completed'
                AND mp.score IS NULL
                AND jr.result->>'episode_id' IS NOT NULL
        """)
        result = await session.execute(matches_query, {"season_id": self.season_id})
        rows = result.all()

        if not rows:
            return False

        match_ids = [row[0] for row in rows]
        episode_ids = [row[1] for row in rows]
        episode_by_match: dict[UUID, str] = {row[0]: row[1] for row in rows}

        # Load matches to get assignments for computing agent counts
        matches = list((await session.execute(select(Match).where(col(Match.id).in_(match_ids)))).scalars().all())
        matches_by_id = {m.id: m for m in matches}

        # Debug: query episode_policies to see num_agents per policy
        ep_result = await session.execute(
            select(
                col(Episode.id).label("episode_id"),
                col(EpisodePolicy.policy_version_id).label("policy_version_id"),
                col(EpisodePolicy.num_agents).label("num_agents"),
            )
            .join(EpisodePolicy, EpisodePolicy.episode_id == Episode.id)  # type: ignore[arg-type]
            .where(col(Episode.id).in_(episode_ids))
            .order_by(col(Episode.id), col(EpisodePolicy.policy_version_id))
        )
        ep_rows = ep_result.all()
        if ep_rows:
            logger.info("Episode policies for score sync:")
            for row in ep_rows:
                logger.info(f"  episode={row.episode_id}, pv={row.policy_version_id}, num_agents={row.num_agents}")

        scores_result = await session.execute(
            select(
                col(Episode.id).label("episode_id"),
                col(PolicyVersion.id).label("policy_version_id"),
                col(EpisodePolicyMetric.value).label("reward"),
            )
            .join(EpisodePolicyMetric, EpisodePolicyMetric.episode_internal_id == Episode.internal_id)  # type: ignore[arg-type]
            .join(PolicyVersion, PolicyVersion.internal_id == EpisodePolicyMetric.pv_internal_id)  # type: ignore[arg-type]
            .where(EpisodePolicyMetric.metric_name == "reward")
            .where(col(Episode.id).in_(episode_ids))
        )
        score_rows = scores_result.all()

        scores_by_episode: dict[str, dict[UUID, float]] = defaultdict(dict)
        for row in score_rows:
            ep_id = str(row.episode_id) if row.episode_id else None
            pv_id = row.policy_version_id
            if ep_id and pv_id:
                scores_by_episode[ep_id][pv_id] = row.reward
                logger.info(f"Policy reward: episode={ep_id}, pv={pv_id}, total_reward={row.reward}")

        if not scores_by_episode:
            return False

        match_players = list(
            (
                await session.execute(
                    select(MatchPlayer)
                    .where(col(MatchPlayer.match_id).in_(match_ids))
                    .options(selectinload(MatchPlayer.pool_player))  # type: ignore[arg-type]
                )
            )
            .scalars()
            .all()
        )

        updated = 0
        for mp in match_players:
            episode_id = episode_by_match.get(mp.match_id)
            if not episode_id:
                continue
            episode_scores = scores_by_episode.get(episode_id, {})
            pv_id = mp.pool_player.policy_version_id
            if pv_id in episode_scores:
                total_reward = episode_scores[pv_id]
                match = matches_by_id.get(mp.match_id)
                agent_count = match.assignments.count(mp.policy_index) if match and match.assignments else 1
                mp.score = total_reward / max(agent_count, 1)
                logger.info(
                    f"Score calc: match={mp.match_id}, pv={pv_id}, "
                    f"total_reward={total_reward}, agent_count={agent_count}, score={mp.score}"
                )
                updated += 1

        if updated > 0:
            logger.info(f"Updated {updated} match player scores")
        await session.commit()
        return updated > 0

    async def _get_pool_players(self, pool_id: UUID) -> list[PoolPlayer]:
        session = get_db()
        return list(
            (await session.execute(select(PoolPlayer).filter_by(pool_id=pool_id, retired=False))).scalars().all()
        )

    async def _get_season_matches(self) -> list[MatchData]:
        session = get_db()
        matches = list(
            (
                await session.execute(
                    select(Match)
                    .join(Pool, Match.pool_id == Pool.id)  # type: ignore[arg-type]
                    .where(Pool.season_id == self.season_id)
                    .options(selectinload(Match.players))  # type: ignore[arg-type]
                )
            )
            .scalars()
            .all()
        )

        result = []
        for m in matches:
            sorted_players = sorted(m.players, key=lambda x: x.policy_index)
            result.append(
                MatchData(
                    match_id=m.id,
                    pool_id=m.pool_id,
                    status=m.status,
                    pool_player_ids=[mp.pool_player_id for mp in sorted_players],
                    assignments=m.assignments or [],
                )
            )
        return result

    @with_db
    async def submit(self, policy_version_id: UUID) -> list[str]:
        changes = self.get_new_submission_membership_changes(policy_version_id)
        pools = await self._ensure_pools_exist(list(self.referees.keys()))
        await self._apply_membership_changes(changes, pools)
        return [c.pool_name for c in changes if c.action == "add"]

    @with_db
    async def get_leaderboard(self) -> list[tuple[UUID, float, int]]:
        session = get_db()
        pool = (
            await session.execute(
                select(Pool)
                .join(Season, Pool.season_id == Season.id)  # type: ignore[arg-type]
                .where(Season.name == self.season_name)
                .where(Pool.name == self.leaderboard_pool)
            )
        ).scalar_one_or_none()
        if not pool:
            return []

        referee = self.referees[self.leaderboard_pool]
        return await referee.get_leaderboard(pool.id)

    @with_db
    async def get_matches(self, pool_name: str, limit: int = 50, offset: int = 0) -> list[Match]:
        session = get_db()
        pool = (
            await session.execute(
                select(Pool)
                .join(Season, Pool.season_id == Season.id)  # type: ignore[arg-type]
                .where(Season.name == self.season_name)
                .where(Pool.name == pool_name)
            )
        ).scalar_one_or_none()
        if not pool:
            raise ValueError(f"Pool '{pool_name}' not found")

        matches = (
            (
                await session.execute(
                    select(Match)
                    .filter_by(pool_id=pool.id)
                    .options(selectinload(Match.players))  # type: ignore[arg-type]
                    .order_by(col(Match.created_at).desc())
                    .offset(offset)
                    .limit(limit)
                )
            )
            .scalars()
            .all()
        )
        return list(matches)

    async def _apply_membership_changes(self, changes: list[MembershipChangeRequest], pools: dict[str, Pool]) -> None:
        if not changes:
            return
        session = get_db()

        all_pool_ids = {pools[c.pool_name].id for c in changes if c.pool_name in pools}
        all_pv_ids = {c.policy_version_id for c in changes if c.pool_name in pools}

        existing_players: dict[tuple[UUID, UUID], PoolPlayer] = {}
        if all_pool_ids and all_pv_ids:
            result = await session.execute(
                select(PoolPlayer)
                .where(col(PoolPlayer.pool_id).in_(all_pool_ids))
                .where(col(PoolPlayer.policy_version_id).in_(all_pv_ids))
            )
            for player in result.scalars().all():
                existing_players[(player.pool_id, player.policy_version_id)] = player

        for change in changes:
            if change.pool_name not in pools:
                continue
            pool = pools[change.pool_name]
            key = (pool.id, change.policy_version_id)

            if change.action == "add":
                if key in existing_players:
                    continue
                player = PoolPlayer(pool_id=pool.id, policy_version_id=change.policy_version_id)
                session.add(player)
                await session.flush()
                session.add(
                    MembershipChange(
                        pool_player_id=player.id,
                        action=MembershipAction.add,
                        notes=change.notes,
                    )
                )
                existing_players[key] = player
                logger.info(f"Added {change.policy_version_id} to pool '{change.pool_name}'")

            elif change.action == "remove":
                player = existing_players.get(key)
                if player and not player.retired:
                    player.retired = True
                    session.add(
                        MembershipChange(
                            pool_player_id=player.id,
                            action=MembershipAction.remove,
                            notes=change.notes,
                        )
                    )
                    logger.info(f"Retired {change.policy_version_id} from pool '{change.pool_name}'")

        await session.commit()

    async def _create_and_dispatch_match(self, pool_id: UUID, request: MatchRequest) -> None:
        session = get_db()

        pool_players = {
            pp.id: pp
            for pp in (await session.execute(select(PoolPlayer).where(col(PoolPlayer.id).in_(request.pool_player_ids))))
            .scalars()
            .all()
        }

        match = Match(pool_id=pool_id, assignments=request.assignments)  # type: ignore[call-arg]
        session.add(match)
        await session.flush()
        session.add_all(
            [
                MatchPlayer(match_id=match.id, pool_player_id=pp_id, policy_index=idx)
                for idx, pp_id in enumerate(request.pool_player_ids)
            ]
        )
        await session.commit()
        match_id, match_pool_id = match.id, match.pool_id

        policy_version_ids = [pool_players[pp_id].policy_version_id for pp_id in request.pool_player_ids]
        policy_uris = [f"metta://policy/{pv}" for pv in policy_version_ids]
        episode_tags = {"match_id": str(match_id), "pool_id": str(match_pool_id), **request.episode_tags}
        job_spec = SingleEpisodeJob(
            policy_uris=policy_uris,
            assignments=request.assignments,
            env=request.env,
            replay_uri=f"{SOFTMAX_S3_REPLAYS_PREFIX}/{match_id}.json.z",
            seed=hash(str(match_id)) % (2**31),
            episode_tags=episode_tags,
        )

        stats_client = StatsClient(settings.STATS_SERVER_URI)
        try:
            job_ids = stats_client.create_jobs([JobRequestCreate(job_type=JobType.episode, job=job_spec.model_dump())])
            job_id = job_ids[0] if job_ids else None
        finally:
            stats_client.close()

        if not job_id:
            logger.error(f"Failed to create job for match {match_id}")
            return

        match = (await session.execute(select(Match).filter_by(id=match_id))).scalar_one()
        match.job_id = job_id
        match.status = MatchStatus.scheduled
        await session.commit()

        logger.debug(f"Match {match_id} -> job {job_id}")
