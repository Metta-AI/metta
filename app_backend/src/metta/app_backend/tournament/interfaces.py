import asyncio
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import UTC, datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel
from sqlalchemy import text
from sqlmodel import col, select

from metta.app_backend.clients.stats_client import StatsClient
from metta.app_backend.database import get_db, with_db
from metta.app_backend.models.job_request import JobRequest, JobRequestCreate, JobStatus, JobType, SingleEpisodeJob
from metta.app_backend.models.tournament import (
    Match,
    MatchPlayer,
    MatchStatus,
    MembershipAction,
    MembershipChangeRecord,
    Pool,
    PoolPlayer,
    Season,
)
from metta.app_backend.tournament.settings import MAX_MATCHES_PER_CYCLE, POLL_INTERVAL_SECONDS, settings
from mettagrid.config.mettagrid_config import MettaGridConfig

logger = logging.getLogger(__name__)

SOFTMAX_S3_REPLAYS_PREFIX = "s3://softmax-public/replays/tournament"


class MatchData(BaseModel):
    match_id: UUID
    status: MatchStatus
    player_pv_ids: list[UUID]


class MatchRequest(BaseModel):
    policy_version_ids: list[UUID]
    assignments: list[int]
    env: MettaGridConfig
    episode_tags: dict[str, str] = {}


class MembershipChange(BaseModel):
    pool_name: str
    policy_version_id: UUID
    action: Literal["add", "retire"]


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


class RefereeInterface(ABC):
    @abstractmethod
    def get_matches_to_schedule(
        self,
        players: list[PoolPlayer],
        matches: list[MatchData],
    ) -> list[MatchRequest]:
        pass


class CommissionerInterface(ABC):
    scorer: ScorerInterface

    season_name: str
    referees: dict[str, RefereeInterface]

    @abstractmethod
    def get_new_submission_membership_changes(self, policy_version_id: UUID) -> list[MembershipChange]:
        pass

    @abstractmethod
    async def get_membership_changes(self, pools: dict[str, Pool]) -> list[MembershipChange]:
        pass

    async def run(self) -> None:
        await self._ensure_season_exists()
        logger.info(f"Starting commissioner for season '{self.season_name}' ({self.season_id})")
        while True:
            try:
                await self._run_cycle()
            except Exception as e:
                logger.error(f"Commissioner cycle error: {e}", exc_info=True)
            await asyncio.sleep(POLL_INTERVAL_SECONDS)

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
    async def _run_cycle(self) -> None:
        pools = await self._ensure_pools_exist(list(self.referees.keys()))

        await self._sync_match_statuses()

        total_scheduled = 0
        for pool_name, pool in pools.items():
            assert pool_name in self.referees
            referee = self.referees[pool_name]
            players = await self._get_pool_players(pool.id)
            matches = await self._get_pool_matches(pool.id)

            requests = referee.get_matches_to_schedule(players, matches)
            for req in requests[:MAX_MATCHES_PER_CYCLE]:
                await self._create_and_dispatch_match(pool.id, req)
                total_scheduled += 1

        if total_scheduled > 0:
            logger.info(f"Scheduled {total_scheduled} new matches")

        changes = await self.get_membership_changes(pools)
        await self._apply_membership_changes(changes, pools)

    async def _ensure_pools_exist(self, pool_names: list[str]) -> dict[str, Pool]:
        session = get_db()
        pools: dict[str, Pool] = {}

        existing_pools = (await session.execute(select(Pool).filter_by(season_id=self.season_id))).scalars().all()
        existing = {p.name: p for p in existing_pools if p.name}

        for name in pool_names:
            if name in existing:
                pools[name] = existing[name]
            else:
                pool = Pool(season_id=self.season_id, name=name)
                session.add(pool)
                await session.flush()
                pools[name] = pool
                logger.info(f"Created pool '{name}' for season {self.season_id}")

        await session.commit()
        return pools

    async def _sync_match_statuses(self) -> None:
        session = get_db()
        pending = list(
            (
                await session.execute(
                    select(Match)
                    .join(Pool, Match.pool_id == Pool.id)  # type: ignore[arg-type]
                    .where(Pool.season_id == self.season_id)
                    .where(col(Match.status).in_([MatchStatus.scheduled, MatchStatus.running]))
                )
            )
            .scalars()
            .all()
        )

        job_ids = [m.job_id for m in pending if m.job_id]
        if not job_ids:
            return

        jobs_by_id = {
            j.id: j
            for j in (await session.execute(select(JobRequest).where(col(JobRequest.id).in_(job_ids)))).scalars()
        }

        updated = 0
        completed_match_ids: list[UUID] = []
        for match in pending:
            if not match.job_id or match.job_id not in jobs_by_id:
                continue
            job = jobs_by_id[match.job_id]
            if job.status == JobStatus.running and match.status != MatchStatus.running:
                match.status = MatchStatus.running
                updated += 1
            elif job.status == JobStatus.completed:
                match.status = MatchStatus.completed
                match.completed_at = datetime.now(UTC)
                completed_match_ids.append(match.id)
                updated += 1
            elif job.status == JobStatus.failed:
                match.status = MatchStatus.failed
                updated += 1

        if updated > 0:
            logger.info(f"Updated {updated} match statuses")
        await session.commit()

        if completed_match_ids:
            await self._sync_match_scores(completed_match_ids)

    async def _sync_match_scores(self, match_ids: list[UUID]) -> None:
        session = get_db()
        scores_query = """
            SELECT et.value AS match_id, pv.id AS policy_version_id, epm.value AS avg_reward
            FROM episode_tags et
            JOIN episodes e ON e.id = et.episode_id
            JOIN episode_policies ep ON ep.episode_id = e.id
            JOIN policy_versions pv ON pv.id = ep.policy_version_id
            JOIN episode_policy_metrics epm ON epm.episode_internal_id = e.internal_id
                AND epm.pv_internal_id = pv.internal_id
                AND epm.metric_name = 'avg_reward'
            WHERE et.key = 'match_id' AND et.value = ANY(:match_ids)
        """
        result = await session.execute(text(scores_query), {"match_ids": [str(m) for m in match_ids]})
        scores_by_match: dict[UUID, dict[UUID, float]] = defaultdict(dict)
        for row in result.all():
            match_id = UUID(row[0])
            pv_id = row[1]
            avg_reward = row[2]
            scores_by_match[match_id][pv_id] = avg_reward

        if not scores_by_match:
            return

        match_players = list(
            (await session.execute(select(MatchPlayer).where(col(MatchPlayer.match_id).in_(match_ids)))).scalars().all()
        )

        updated = 0
        for mp in match_players:
            match_scores = scores_by_match.get(mp.match_id, {})
            if mp.policy_version_id in match_scores:
                mp.score = match_scores[mp.policy_version_id]
                updated += 1

        if updated > 0:
            logger.info(f"Updated {updated} match player scores")
        await session.commit()

    async def _get_pool_players(self, pool_id: UUID) -> list[PoolPlayer]:
        session = get_db()
        return list(
            (await session.execute(select(PoolPlayer).filter_by(pool_id=pool_id, retired=False))).scalars().all()
        )

    async def _get_pool_matches(self, pool_id: UUID) -> list[MatchData]:
        session = get_db()
        matches = list((await session.execute(select(Match).filter_by(pool_id=pool_id))).scalars().all())

        match_ids = [m.id for m in matches]
        if not match_ids:
            return []

        match_players = list(
            (await session.execute(select(MatchPlayer).where(col(MatchPlayer.match_id).in_(match_ids)))).scalars().all()
        )

        players_by_match: dict[UUID, list[UUID]] = defaultdict(list)
        for mp in match_players:
            players_by_match[mp.match_id].append(mp.policy_version_id)

        return [MatchData(match_id=m.id, status=m.status, player_pv_ids=players_by_match[m.id]) for m in matches]

    @with_db
    async def submit(self, policy_version_id: UUID) -> list[str]:
        changes = self.get_new_submission_membership_changes(policy_version_id)
        adds = [c for c in changes if c.action == "add"]
        if not adds:
            raise ValueError("No initial pools configured")

        session = get_db()
        season = (await session.execute(select(Season).filter_by(name=self.season_name))).scalar_one_or_none()
        if not season:
            raise ValueError(f"Season '{self.season_name}' not initialized")

        pool_names = [c.pool_name for c in adds]
        pools = {
            p.name: p
            for p in (
                await session.execute(select(Pool).filter_by(season_id=season.id).where(col(Pool.name).in_(pool_names)))
            )
            .scalars()
            .all()
        }

        for pool_name in pool_names:
            if pool_name not in pools:
                raise ValueError(f"Pool '{pool_name}' not found")

        for change in adds:
            pool = pools[change.pool_name]
            existing = (
                await session.execute(
                    select(PoolPlayer).filter_by(pool_id=pool.id, policy_version_id=policy_version_id, retired=False)
                )
            ).scalar_one_or_none()
            if existing:
                raise ValueError(f"Already submitted to pool '{change.pool_name}'")

        session.add_all([PoolPlayer(pool_id=pools[c.pool_name].id, policy_version_id=policy_version_id) for c in adds])
        await session.commit()
        logger.info(f"Submitted {policy_version_id} to pools {pool_names} in season '{self.season_name}'")
        return pool_names

    @with_db
    async def get_leaderboard(self, pool_name: str | None = None) -> list[tuple[UUID, float]]:
        session = get_db()
        season = (await session.execute(select(Season).filter_by(name=self.season_name))).scalar_one_or_none()
        if not season:
            return []

        if pool_name:
            pool = (
                await session.execute(select(Pool).filter_by(season_id=season.id, name=pool_name))
            ).scalar_one_or_none()
            if not pool:
                raise ValueError(f"Pool '{pool_name}' not found")
            pool_ids = [pool.id]
        else:
            pool_ids = [row[0] for row in (await session.execute(select(Pool.id).filter_by(season_id=season.id))).all()]

        if not pool_ids:
            return []

        matches = list(
            (
                await session.execute(
                    select(Match).where(col(Match.pool_id).in_(pool_ids)).where(Match.status == MatchStatus.completed)
                )
            )
            .scalars()
            .all()
        )

        if not matches:
            return []

        all_policy_ids: set[UUID] = set()
        scored_matches: list[ScoredMatchData] = []

        for match in matches:
            if not match.players or any(mp.score is None for mp in match.players):
                continue

            policy_scores: dict[UUID, float] = {}
            policy_version_ids: list[UUID] = []
            for mp in sorted(match.players, key=lambda x: x.policy_index):
                policy_scores[mp.policy_version_id] = mp.score  # type: ignore[assignment]
                if mp.policy_index >= len(policy_version_ids):
                    policy_version_ids.append(mp.policy_version_id)
                all_policy_ids.add(mp.policy_version_id)

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
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    @with_db
    async def get_matches(self, pool_name: str, limit: int = 50, offset: int = 0) -> list[Match]:
        session = get_db()
        season = (await session.execute(select(Season).filter_by(name=self.season_name))).scalar_one_or_none()
        if not season:
            return []

        pool = (await session.execute(select(Pool).filter_by(season_id=season.id, name=pool_name))).scalar_one_or_none()
        if not pool:
            raise ValueError(f"Pool '{pool_name}' not found")

        matches = (
            (
                await session.execute(
                    select(Match)
                    .filter_by(pool_id=pool.id)
                    .order_by(col(Match.created_at).desc())
                    .offset(offset)
                    .limit(limit)
                )
            )
            .scalars()
            .all()
        )
        return list(matches)

    async def _apply_membership_changes(self, changes: list[MembershipChange], pools: dict[str, Pool]) -> None:
        if not changes:
            return
        session = get_db()

        adds_by_pool: dict[str, list[UUID]] = defaultdict(list)
        retires: list[tuple[UUID, UUID]] = []

        for change in changes:
            if change.pool_name not in pools:
                continue
            pool = pools[change.pool_name]
            if change.action == "add":
                adds_by_pool[change.pool_name].append(change.policy_version_id)
            elif change.action == "retire":
                retires.append((pool.id, change.policy_version_id))

        records: list[MembershipChangeRecord] = []

        for pool_name, pv_ids in adds_by_pool.items():
            pool = pools[pool_name]
            existing = set(
                (
                    await session.execute(
                        select(PoolPlayer.policy_version_id)
                        .filter_by(pool_id=pool.id)
                        .where(col(PoolPlayer.policy_version_id).in_(pv_ids))
                    )
                )
                .scalars()
                .all()
            )
            new_pv_ids = [pv for pv in pv_ids if pv not in existing]
            if new_pv_ids:
                session.add_all([PoolPlayer(pool_id=pool.id, policy_version_id=pv_id) for pv_id in new_pv_ids])
                records.extend(
                    [
                        MembershipChangeRecord(pool_id=pool.id, policy_version_id=pv_id, action=MembershipAction.add)
                        for pv_id in new_pv_ids
                    ]
                )
                logger.info(f"Added {len(new_pv_ids)} players to pool '{pool_name}'")

        for pool_id, pv_id in retires:
            result = await session.execute(
                select(PoolPlayer).filter_by(pool_id=pool_id, policy_version_id=pv_id, retired=False)
            )
            player = result.scalar_one_or_none()
            if player:
                player.retired = True
                records.append(
                    MembershipChangeRecord(pool_id=pool_id, policy_version_id=pv_id, action=MembershipAction.retire)
                )

        if records:
            session.add_all(records)
            await session.commit()

    async def _create_and_dispatch_match(self, pool_id: UUID, request: MatchRequest) -> None:
        session = get_db()
        match = Match(pool_id=pool_id, assignments=request.assignments)  # type: ignore[call-arg]
        session.add(match)
        await session.flush()
        session.add_all(
            [
                MatchPlayer(match_id=match.id, policy_version_id=pv_id, policy_index=idx)
                for idx, pv_id in enumerate(request.policy_version_ids)
            ]
        )
        await session.commit()
        match_id, match_pool_id = match.id, match.pool_id

        policy_uris = [f"metta://policy/{pv}" for pv in request.policy_version_ids]
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
