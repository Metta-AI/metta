import asyncio
import logging
import uuid
from datetime import datetime
from typing import Optional

from psycopg import AsyncConnection
from pydantic import BaseModel

from softmax.orchestrator.metta_repo import LeaderboardRow, MettaRepo


class EvalScore(BaseModel):
    eval_name: str
    total_score: float
    num_agents: int

    @property
    def score(self) -> float:
        return self.total_score / self.num_agents


logger = logging.getLogger("leaderboard_updater")


class LeaderboardUpdater:
    def __init__(self, repo: MettaRepo):
        self.repo = repo
        self.task: Optional[asyncio.Task] = None
        self.running = False

    async def start(self):
        """Start the leaderboard updater as a background task."""
        self.running = True
        self.task = asyncio.create_task(self._run_loop(), name="LeaderboardUpdater")
        logger.info("Leaderboard updater started")

    async def stop(self):
        """Stop the leaderboard updater and wait for completion."""
        self.running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        logger.info("Leaderboard updater stopped")

    async def _run_loop(self):
        """Main loop that runs the updater periodically."""
        while self.running:
            try:
                await self._run_once()
            except Exception as e:
                logger.error(f"Error updating leaderboards: {e}")

            # Non-blocking sleep using asyncio
            await asyncio.sleep(10)

    async def _get_policy_scores_batch(
        self, policy_ids: list[uuid.UUID], eval_names: list[str], metric: str, latest_episode_id: int
    ) -> dict[uuid.UUID, float]:
        """Get scores for multiple policies in a single batch query.

        The score is computed as the average of the scores of each eval, unweighted by the number of episodes.
        """
        if not policy_ids:
            return {}

        query = """
                  WITH policy_eval_scores AS (
                    SELECT e.primary_policy_id AS policy_id, e.eval_name, AVG(eam.value) AS score
                    FROM episodes e
                    JOIN episode_agent_metrics eam ON e.internal_id = eam.episode_internal_id
                    WHERE e.primary_policy_id = ANY(%s)
                    AND eam.metric = %s
                    AND e.internal_id <= %s
                    AND e.eval_name = ANY(%s)
                    GROUP BY e.primary_policy_id, e.eval_name
                  )
                  SELECT policy_id, SUM(score) / %s as score
                  FROM policy_eval_scores
                  GROUP BY policy_id
                  """

        async with self.repo.connect() as con:
            async with con.cursor() as cursor:
                await cursor.execute(query, (policy_ids, metric, latest_episode_id, eval_names, len(eval_names)))
                rows = await cursor.fetchall()

                return {row[0]: row[1] for row in rows}

    async def _get_updated_policies(self, leaderboard: LeaderboardRow, latest_episode_id: int) -> list[uuid.UUID]:
        """Get the policies that have had new relevant episodes."""

        async with self.repo.connect() as con:
            async with con.cursor() as cursor:
                res = await cursor.execute(
                    """
                            SELECT DISTINCT p.id
                            FROM
                              policies p
                              LEFT JOIN epochs e ON p.epoch_id = e.id
                              LEFT JOIN training_runs tr ON e.run_id = tr.id
                              JOIN episodes ep ON ep.primary_policy_id = p.id
                            WHERE ep.internal_id > %s
                            AND ep.internal_id <= %s
                            AND COALESCE(tr.created_at, p.created_at) >= %s
                            AND ep.eval_name = ANY(%s)
                          """,
                    (leaderboard.latest_episode, latest_episode_id, leaderboard.start_date, leaderboard.evals),
                )
                rows = await res.fetchall()
                return [row[0] for row in rows]

    async def _get_latest_episode_id(self) -> int:
        async with self.repo.connect() as con:
            async with con.cursor() as cursor:
                await cursor.execute("SELECT MAX(internal_id) FROM episodes", ())
                latest_episode_id_row = await cursor.fetchone()
                if latest_episode_id_row is None:
                    return 0
                return latest_episode_id_row[0] or 0

    async def _check_leaderboard_consistency(
        self, con: AsyncConnection, leaderboard_id: uuid.UUID, updated_at: datetime
    ) -> None:
        """
        If the leaderboard was updated during processing, throw an exception which will rollback this
        transaction, and also exit the processing of this leaderboard. The leaderboard will be re-processed next time.
        """

        cur_updated_at = await con.execute("SELECT updated_at FROM leaderboards WHERE id = %s", (leaderboard_id,))
        cur_updated_at_row = await cur_updated_at.fetchone()
        if cur_updated_at_row is None:
            raise RuntimeError(f"Leaderboard {leaderboard_id} not found")
        if cur_updated_at_row[0] != updated_at:
            raise RuntimeError(f"Leaderboard {leaderboard_id} was updated during processing")

    async def _batch_upsert_leaderboard_policy_scores(
        self, leaderboard_id: uuid.UUID, policy_scores: dict[uuid.UUID, float], updated_at: datetime
    ) -> None:
        """Batch upsert leaderboard policy scores for multiple policies, chunked to avoid overwhelming the system."""
        if not policy_scores:
            return

        # Prepare all batch data
        all_batch_data = [(leaderboard_id, policy_id, score, score) for policy_id, score in policy_scores.items()]

        # Process in chunks
        async with self.repo.connect() as con:
            async with con.cursor() as cursor:
                await cursor.executemany(
                    """
                    INSERT INTO leaderboard_policy_scores (leaderboard_id, policy_id, score) VALUES (%s, %s, %s)
                    ON CONFLICT (leaderboard_id, policy_id) DO UPDATE SET score = %s
                    """,
                    all_batch_data,
                )

            await self._check_leaderboard_consistency(con, leaderboard_id, updated_at)

    async def _update_leaderboard(self, leaderboard: LeaderboardRow, chunk_size: int = 5000):
        """This function maintains the (leaderboard_id, policy_id) -> score mapping in the leaderboard_policy_scores
        table.

        It does this by:
        - Getting the latest episode ID
        - Getting the policies that have had new relevant episodes
        - Re-computing the score for each policy and updating the leaderboard_policy_scores table
        - Updating the leaderboard latest_episode_id column
        """

        latest_episode_id = await self._get_latest_episode_id()
        if latest_episode_id == leaderboard.latest_episode:
            return
        if latest_episode_id < leaderboard.latest_episode:
            raise ValueError(
                f"Latest episode ID {latest_episode_id} < leaderboard latest episode {leaderboard.latest_episode}"
            )

        updated_policies = await self._get_updated_policies(leaderboard, latest_episode_id)
        if len(updated_policies) == 0:
            logger.info(f"No updated policies for leaderboard {leaderboard.id}")
            return

        logger.info(f"Updating leaderboard {leaderboard.id} with {len(updated_policies)} updated policies")

        for i in range(0, len(updated_policies), chunk_size):
            chunk = updated_policies[i : i + chunk_size]
            policy_scores = await self._get_policy_scores_batch(
                chunk, leaderboard.evals, leaderboard.metric, latest_episode_id
            )

            # Batch upsert all scores (chunked for large datasets)
            await self._batch_upsert_leaderboard_policy_scores(leaderboard.id, policy_scores, leaderboard.updated_at)

        async with self.repo.connect() as con:
            await self.repo.update_leaderboard_latest_episode(con, leaderboard.id, latest_episode_id)
            await self._check_leaderboard_consistency(con, leaderboard.id, leaderboard.updated_at)

    async def _run_once(self):
        """Run one iteration of the leaderboard update process."""
        leaderboards = await self.repo.list_leaderboards()
        for leaderboard in leaderboards:
            try:
                await self._update_leaderboard(leaderboard)
            except Exception as e:
                logger.error(f"Error updating leaderboard {leaderboard.id}: {e}")
                # Continue with other leaderboards instead of failing completely
