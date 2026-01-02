import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import col, select

import mettagrid.builder.envs as eb
from metta.app_backend.clients.stats_client import StatsClient
from metta.app_backend.models.job_request import JobRequest, JobRequestCreate, JobStatus, JobType
from metta.app_backend.models.tournament import Match, MatchPlayer, MatchStatus, Pool, PoolPlayer
from metta.app_backend.tournament.settings import settings
from metta.common.util.log_config import init_logging
from metta.sim.single_episode_runner import SingleEpisodeJob

logger = logging.getLogger(__name__)

SOFTMAX_S3_REPLAYS_PREFIX = "s3://softmax-public/replays/tournament"

_engine = None
_session_factory = None


def _get_session_factory() -> async_sessionmaker[AsyncSession]:
    global _engine, _session_factory
    if _session_factory is None:
        db_uri = settings.STATS_DB_URI
        if db_uri.startswith("postgres://"):
            db_uri = "postgresql+psycopg_async://" + db_uri.split("://", 1)[1]
        elif db_uri.startswith("postgresql://"):
            db_uri = "postgresql+psycopg_async://" + db_uri.split("://", 1)[1]
        _engine = create_async_engine(db_uri, pool_size=5, max_overflow=10)
        _session_factory = async_sessionmaker(_engine, class_=AsyncSession, expire_on_commit=False)
    return _session_factory


@dataclass
class MatchData:
    match: Match
    player_pv_ids: list[UUID]


@dataclass
class MatchRequest:
    policy_version_ids: list[UUID]
    assignments: list[int]


class RefereeInterface(ABC):
    @abstractmethod
    def get_matches_to_schedule(
        self,
        player_pv_ids: list[UUID],
        matches: list[MatchData],
    ) -> list[MatchRequest]:
        pass


class TwoStageReferee(RefereeInterface):
    def __init__(
        self,
        selfplay_matches: int = settings.SELFPLAY_MATCHES,
        top_k: int = settings.TOP_K,
        pair_matches: int = settings.PAIR_MATCHES,
        max_per_cycle: int = settings.MAX_MATCHES_PER_CYCLE,
    ):
        self.selfplay_matches = selfplay_matches
        self.top_k = top_k
        self.pair_matches = pair_matches
        self.max_per_cycle = max_per_cycle

    def get_matches_to_schedule(
        self,
        player_pv_ids: list[UUID],
        matches: list[MatchData],
    ) -> list[MatchRequest]:
        requests: list[MatchRequest] = []

        selfplay_counts: dict[UUID, int] = {pv: 0 for pv in player_pv_ids}
        pair_counts: dict[tuple[UUID, UUID], int] = {}

        for md in matches:
            pv_set = set(md.player_pv_ids)
            if len(pv_set) == 1:
                pv = md.player_pv_ids[0]
                selfplay_counts[pv] = selfplay_counts.get(pv, 0) + 1
            elif len(pv_set) == 2:
                pair = tuple(sorted(pv_set))
                pair_counts[pair] = pair_counts.get(pair, 0) + 1

        # Stage 1: Self-play
        for pv in player_pv_ids:
            if len(requests) >= self.max_per_cycle:
                break
            needed = self.selfplay_matches - selfplay_counts.get(pv, 0)
            for _ in range(min(needed, self.max_per_cycle - len(requests))):
                requests.append(MatchRequest(policy_version_ids=[pv], assignments=[0, 0, 0, 0]))

        # Stage 2: Pair top-k once all self-play done
        all_selfplay_done = all(selfplay_counts.get(pv, 0) >= self.selfplay_matches for pv in player_pv_ids)
        if all_selfplay_done and len(player_pv_ids) >= 2:
            top_policies = player_pv_ids[: self.top_k]
            for i, pv1 in enumerate(top_policies):
                if len(requests) >= self.max_per_cycle:
                    break
                for pv2 in top_policies[i + 1 :]:
                    if len(requests) >= self.max_per_cycle:
                        break
                    pair = tuple(sorted([pv1, pv2]))
                    needed = self.pair_matches - pair_counts.get(pair, 0)
                    for _ in range(min(needed, self.max_per_cycle - len(requests))):
                        requests.append(MatchRequest(policy_version_ids=[pv1, pv2], assignments=[0, 0, 1, 1]))

        return requests


async def run_referee(referee: RefereeInterface | None = None):
    if referee is None:
        referee = TwoStageReferee()

    logger.info(f"Starting referee (poll={settings.POLL_INTERVAL_SECONDS}s)")
    while True:
        try:
            await _run_cycle(referee)
        except Exception as e:
            logger.error(f"Referee cycle error: {e}", exc_info=True)
        await asyncio.sleep(settings.POLL_INTERVAL_SECONDS)


async def _run_cycle(referee: RefereeInterface):
    factory = _get_session_factory()
    async with factory() as session:
        pools = list((await session.execute(select(Pool).where(Pool.season_id.is_not(None)))).scalars().all())

        pending = list(
            (
                await session.execute(
                    select(Match).where(col(Match.status).in_([MatchStatus.scheduled, MatchStatus.running]))
                )
            )
            .scalars()
            .all()
        )

        job_ids = [m.job_id for m in pending if m.job_id]
        jobs_by_id: dict[UUID, JobRequest] = {}
        if job_ids:
            job_result = await session.execute(select(JobRequest).where(col(JobRequest.id).in_(job_ids)))
            jobs_by_id = {j.id: j for j in job_result.scalars()}

        for match in pending:
            if not match.job_id or match.job_id not in jobs_by_id:
                continue
            job = jobs_by_id[match.job_id]
            if job.status == JobStatus.running and match.status != MatchStatus.running:
                match.status = MatchStatus.running
            elif job.status == JobStatus.completed:
                match.status = MatchStatus.completed
                match.completed_at = datetime.now(UTC)
            elif job.status == JobStatus.failed:
                match.status = MatchStatus.failed

        await session.commit()

    for pool in pools:
        try:
            await _schedule_pool_matches(referee, pool)
        except Exception as e:
            logger.error(f"Pool {pool.id} error: {e}", exc_info=True)


async def _schedule_pool_matches(referee: RefereeInterface, pool: Pool):
    factory = _get_session_factory()
    async with factory() as session:
        players = list(
            (
                await session.execute(
                    select(PoolPlayer).where(PoolPlayer.pool_id == pool.id, PoolPlayer.retired == False)
                )
            )  # noqa
            .scalars()
            .all()
        )
        player_pv_ids = [p.policy_version_id for p in players]

        matches = list((await session.execute(select(Match).where(Match.pool_id == pool.id))).scalars().all())
        match_ids = [m.id for m in matches]

        match_players: list[MatchPlayer] = []
        if match_ids:
            mp_result = await session.execute(select(MatchPlayer).where(col(MatchPlayer.match_id).in_(match_ids)))
            match_players = list(mp_result.scalars().all())

        players_by_match: dict[UUID, list[UUID]] = {}
        for mp in match_players:
            players_by_match.setdefault(mp.match_id, []).append(mp.policy_version_id)

        match_data = [MatchData(match=m, player_pv_ids=players_by_match.get(m.id, [])) for m in matches]

    requests = referee.get_matches_to_schedule(player_pv_ids, match_data)
    for req in requests:
        await _create_and_dispatch_match(pool.id, req)


async def _create_and_dispatch_match(pool_id: UUID, request: MatchRequest):
    factory = _get_session_factory()
    async with factory() as session:
        match = Match(pool_id=pool_id, environment_name="arena", assignments=request.assignments)
        session.add(match)
        await session.flush()
        for idx, pv_id in enumerate(request.policy_version_ids):
            session.add(MatchPlayer(match_id=match.id, policy_version_id=pv_id, policy_index=idx))
        await session.commit()
        match_id, match_pool_id = match.id, match.pool_id

    policy_uris = [f"metta://policy/{pv}" for pv in request.policy_version_ids]
    job_spec = SingleEpisodeJob(
        policy_uris=policy_uris,
        assignments=request.assignments,
        env=eb.make_arena(num_agents=4, combat=True),
        replay_uri=f"{SOFTMAX_S3_REPLAYS_PREFIX}/{match_id}.json.z",
        seed=hash(str(match_id)) % (2**31),
        episode_tags={"match_id": str(match_id), "pool_id": str(match_pool_id)},
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

    async with factory() as session:
        match = (await session.execute(select(Match).where(Match.id == match_id))).scalar_one()
        match.job_id = job_id
        match.status = MatchStatus.scheduled
        await session.commit()

    logger.info(f"Match {match_id} -> job {job_id}")


if __name__ == "__main__":


    init_logging()
    asyncio.run(run_referee())
