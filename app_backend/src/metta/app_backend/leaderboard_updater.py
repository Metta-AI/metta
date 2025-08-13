# A thread that runs periodically and updates the leaderboard data.
# Uses threads, not asyncio.

import logging
import threading
import time
import uuid

from psycopg.rows import class_row
from pydantic import BaseModel

from metta.app_backend.metta_repo import LeaderboardRow, MettaRepo


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

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.name = "LeaderboardUpdater"
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()

    async def _get_policy_score(
        self, policy_id: uuid.UUID, eval_names: list[str], metric: str, latest_episode_id: int
    ) -> float:
        """Get the score for a policy.

        The score is computed as the average of the scores of each eval, unweighted by the number of episodes.
        """

        query = """
                  SELECT e.eval_name, SUM(eam.value) as total_score, COUNT(*) as num_agents
                  FROM episodes e
                  JOIN episode_agent_metrics eam ON e.internal_id = eam.episode_internal_id
                  WHERE e.primary_policy_id = %s
                  AND eam.metric = %s
                  AND e.internal_id <= %s
                  AND e.eval_name = ANY(%s)
                  GROUP BY e.eval_name
                  """

        async with self.repo.connect() as con:
            async with con.cursor(row_factory=class_row(EvalScore)) as cursor:
                await cursor.execute(query, (policy_id, metric, latest_episode_id, eval_names))

                rows = await cursor.fetchall()
                if len(rows) == 0:
                    return 0
                else:
                    return sum(row.score for row in rows) / len(eval_names)

    async def _get_updated_policies(self, leaderboard: LeaderboardRow, latest_episode_id: int) -> list[uuid.UUID]:
        """Get the policies that have had new relevant episodes."""

        async with self.repo.connect() as con:
            async with con.cursor() as cursor:
                res = await cursor.execute(
                    """
                            SELECT DISTINCT p.id
                            FROM policies p
                            JOIN episodes e ON e.primary_policy_id = p.id
                            WHERE e.internal_id > %s
                            AND e.internal_id <= %s
                            AND p.created_at >= %s
                            AND e.eval_name = ANY(%s)
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
                return latest_episode_id_row[0]

    async def _update_leaderboard(self, leaderboard: LeaderboardRow):
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
            return

        logger.info(f"Updating leaderboard {leaderboard.id} with {len(updated_policies)} updated policies")

        for policy_id in updated_policies:
            policy_score = await self._get_policy_score(
                policy_id, leaderboard.evals, leaderboard.metric, latest_episode_id
            )
            await self.repo.upsert_leaderboard_policy_score(leaderboard.id, policy_id, policy_score)

        await self.repo.update_leaderboard_latest_episode(leaderboard.id, latest_episode_id)

    async def _run_once(self):
        leaderboards = await self.repo.list_leaderboards()
        for leaderboard in leaderboards:
            await self._update_leaderboard(leaderboard)

    def run(self):
        while self.running:
            try:
                import asyncio

                asyncio.run(self._run_once())
            except Exception as e:
                logger.error(f"Error updating leaderboards: {e}")

            time.sleep(30)
