"""Tests for bulk episode upload functionality."""

import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import duckdb
import pytest
import pytest_asyncio
from fastapi.testclient import TestClient

from metta.app_backend.clients.stats_client import StatsClient
from metta.app_backend.episode_stats_db import (
    create_episode_stats_db,
    insert_agent_metric,
    insert_agent_policy,
    insert_episode,
    insert_episode_tag,
)
from metta.app_backend.metta_repo import MettaRepo


class TestBulkEpisodeUpload:
    """Tests for bulk episode upload route and client."""

    def _setup_s3_mocks(self, mock_aioboto3: MagicMock, db_path: Path) -> None:
        """Set up S3 mocks for presigned URL flow."""
        mock_s3_client = AsyncMock()
        mock_s3_client.generate_presigned_url = AsyncMock(return_value="https://s3.amazonaws.com/presigned-url")
        mock_s3_client.head_object = AsyncMock()
        mock_s3_client.download_file = AsyncMock()

        mock_client_context = AsyncMock()
        mock_client_context.__aenter__.return_value = mock_s3_client
        mock_client_context.__aexit__.return_value = None

        mock_session = MagicMock()
        mock_session.client.return_value = mock_client_context
        mock_aioboto3.Session.return_value = mock_session

        # Mock S3 download to copy the test file
        async def mock_download(bucket: str, key: str, filename: str):
            import shutil

            shutil.copy(str(db_path), filename)

        mock_s3_client.download_file.side_effect = mock_download

    def _call_presigned_upload_flow(self, test_client: TestClient, headers: dict[str, str]) -> dict:
        """Execute the presigned URL flow: get URL, then complete upload."""
        # Step 1: Get presigned URL
        presigned_response = test_client.post("/stats/episodes/bulk_upload/presigned-url", headers=headers)
        assert presigned_response.status_code == 200
        presigned_data = presigned_response.json()

        # Step 2: Complete upload (backend downloads from S3 and processes)
        complete_response = test_client.post(
            "/stats/episodes/bulk_upload/complete",
            json={"upload_id": presigned_data["upload_id"]},
            headers=headers,
        )
        assert complete_response.status_code == 200
        return complete_response.json()

    async def _create_policy_version(self, stats_repo: MettaRepo) -> uuid.UUID:
        """Helper to create a policy and policy version in the database."""
        from psycopg.types.json import Jsonb

        async with stats_repo.connect() as con:
            # Create policy with a unique name
            policy_name = f"test_policy_{uuid.uuid4().hex[:8]}"
            result = await con.execute(
                """
                INSERT INTO policies (name, user_id, attributes)
                VALUES (%s, %s, %s)
                RETURNING id
                """,
                (policy_name, "test_user", Jsonb({})),
            )
            row = await result.fetchone()
            assert row is not None
            policy_id = row[0]

            # Create policy version
            result = await con.execute(
                """
                INSERT INTO policy_versions (policy_id, version, policy_spec, attributes)
                VALUES (%s, %s, %s, %s)
                RETURNING id
                """,
                (policy_id, 1, Jsonb({"type": "test"}), Jsonb({})),
            )
            row = await result.fetchone()
            assert row is not None
            return row[0]

    @pytest_asyncio.fixture
    async def sample_duckdb(self, stats_repo: MettaRepo) -> Path:
        """Create a sample DuckDB file with episode stats."""
        # Create policy version in the database
        pv_id = await self._create_policy_version(stats_repo)

        # Create DuckDB with episode stats schema
        conn, db_path = create_episode_stats_db()

        # Insert sample data using the actual policy version ID from the database
        episode_id = str(uuid.uuid4())

        insert_episode(
            conn,
            episode_id=episode_id,
            primary_pv_id=str(pv_id),
            replay_url="https://example.com/replay.json",
            thumbnail_url=None,
            attributes={"suite": "test", "name": "test_env"},
            eval_task_id=None,
        )

        insert_episode_tag(conn, episode_id, "environment", "test_env")

        # Add 4 agents with same policy
        for agent_id in range(4):
            insert_agent_policy(conn, episode_id, str(pv_id), agent_id)

            # Add reward metric for each agent
            insert_agent_metric(conn, episode_id, agent_id, "reward", 1.5 * (agent_id + 1))

        conn.close()
        return db_path

    @pytest.mark.asyncio
    @patch("httpx.put")
    @patch("metta.app_backend.routes.stats_routes.aioboto3")
    async def test_bulk_upload_client_presigned_url(
        self,
        mock_aioboto3: MagicMock,
        mock_httpx_put: MagicMock,
        stats_client: StatsClient,
        sample_duckdb: Path,
        stats_repo: MettaRepo,
    ):
        """Test bulk upload via StatsClient using presigned URL flow."""
        self._setup_s3_mocks(mock_aioboto3, sample_duckdb)

        # Mock httpx PUT for S3 upload
        mock_put_response = MagicMock()
        mock_put_response.raise_for_status = MagicMock()
        mock_httpx_put.return_value = mock_put_response

        # Create a policy and policy version first
        policy_response = stats_client.create_policy(name="test_policy", attributes={}, is_system_policy=False)
        pv_response = stats_client.create_policy_version(
            policy_id=policy_response.id,
            policy_spec={"type": "test"},
            git_hash="abc123",
            attributes={"epoch": 1},
        )

        # Update the DuckDB to use the actual policy version ID
        conn = duckdb.connect(str(sample_duckdb))
        conn.execute(f"UPDATE episodes SET primary_pv_id = '{pv_response.id}'")
        conn.execute(f"UPDATE episode_agent_policies SET policy_version_id = '{pv_response.id}'")
        conn.close()

        # Upload via client
        response = stats_client.bulk_upload_episodes(str(sample_duckdb))

        assert response.episodes_created == 1
        assert "s3://" in response.duckdb_s3_uri

        # Verify data in database
        async with stats_repo.connect() as con:
            result = await con.execute("SELECT COUNT(*) FROM episodes WHERE primary_pv_id = %s", (pv_response.id,))
            row = await result.fetchone()
            assert row is not None
            assert row[0] >= 1

    @pytest.mark.asyncio
    @patch("metta.app_backend.routes.stats_routes.aioboto3")
    async def test_bulk_upload_multiple_episodes(
        self,
        mock_aioboto3: MagicMock,
        test_client: TestClient,
        auth_headers: dict[str, str],
        stats_repo: MettaRepo,
    ):
        """Test uploading multiple episodes at once using presigned URL flow."""
        # Create policy version in the database first
        pv_id = await self._create_policy_version(stats_repo)

        # Create DuckDB with episode stats schema
        conn, db_path = create_episode_stats_db()

        # Insert 3 episodes
        pv_id_str = str(pv_id)
        for ep_idx in range(3):
            episode_id = str(uuid.uuid4())

            insert_episode(
                conn,
                episode_id=episode_id,
                primary_pv_id=pv_id_str,
                replay_url=f"https://example.com/replay_{ep_idx}.json",
                thumbnail_url=None,
                attributes={"episode": ep_idx},
                eval_task_id=None,
            )

            # Add 2 agents per episode
            for agent_id in range(2):
                insert_agent_policy(conn, episode_id, pv_id_str, agent_id)
                insert_agent_metric(conn, episode_id, agent_id, "reward", 2.0 + agent_id)

        conn.close()

        self._setup_s3_mocks(mock_aioboto3, db_path)
        data = self._call_presigned_upload_flow(test_client, auth_headers)

        assert data["episodes_created"] == 3

        # Verify all episodes were created
        async with stats_repo.connect() as con:
            result = await con.execute("SELECT COUNT(*) FROM episodes")
            row = await result.fetchone()
            assert row is not None
            assert row[0] >= 3

    @pytest.mark.asyncio
    @patch("metta.app_backend.routes.stats_routes.aioboto3")
    async def test_metric_aggregation(
        self,
        mock_aioboto3: MagicMock,
        test_client: TestClient,
        auth_headers: dict[str, str],
        stats_repo: MettaRepo,
    ):
        """Test that agent metrics are correctly aggregated to policy metrics using presigned URL flow."""
        # Create two policy versions in the database
        pv1_id = await self._create_policy_version(stats_repo)
        pv2_id = await self._create_policy_version(stats_repo)

        # Create DuckDB with episode stats schema
        conn, db_path = create_episode_stats_db()

        episode_id = str(uuid.uuid4())
        pv1_id_str = str(pv1_id)
        pv2_id_str = str(pv2_id)

        insert_episode(
            conn,
            episode_id=episode_id,
            primary_pv_id=pv1_id_str,
            replay_url=None,
            thumbnail_url=None,
            attributes={},
            eval_task_id=None,
        )

        # Add 6 agents: 4 with policy 1, 2 with policy 2
        for agent_id in range(6):
            pv_id = pv1_id_str if agent_id < 4 else pv2_id_str
            insert_agent_policy(conn, episode_id, pv_id, agent_id)
            # Each agent gets reward of 10.0
            insert_agent_metric(conn, episode_id, agent_id, "reward", 10.0)

        conn.close()

        self._setup_s3_mocks(mock_aioboto3, db_path)
        self._call_presigned_upload_flow(test_client, auth_headers)

        # Verify aggregation
        async with stats_repo.connect() as con:
            # Should have 2 policy entries
            result = await con.execute(
                """
                SELECT policy_version_id, num_agents
                FROM episode_policies
                ORDER BY num_agents DESC
                """
            )
            policies = await result.fetchall()
            assert len(policies) >= 2

            # Check that we have 4 agents for one policy and 2 for another
            agent_counts = sorted([p[1] for p in policies], reverse=True)
            assert 4 in agent_counts[:5]  # Should be in top 5
            assert 2 in agent_counts[:10]  # Should be in top 10

            # Check aggregated metrics
            result = await con.execute(
                """
                SELECT pv_internal_id, metric_name, value
                FROM episode_policy_metrics
                WHERE metric_name = 'reward'
                """
            )
            metrics = await result.fetchall()
            assert len(metrics) >= 2

            # Policy 1 should have 40.0 (4 agents * 10.0)
            # Policy 2 should have 20.0 (2 agents * 10.0)
            values = sorted([m[2] for m in metrics], reverse=True)
            assert any(abs(v - 40.0) < 0.01 for v in values[:5])
            assert any(abs(v - 20.0) < 0.01 for v in values[:10])

    @pytest.mark.asyncio
    @patch("metta.app_backend.routes.stats_routes.aioboto3")
    async def test_non_reward_metrics_filtered(
        self,
        mock_aioboto3: MagicMock,
        test_client: TestClient,
        auth_headers: dict[str, str],
        stats_repo: MettaRepo,
    ):
        """Test that only 'reward' metrics are stored (whitelist) using presigned URL flow."""
        # Create policy version in the database first
        pv_id = await self._create_policy_version(stats_repo)

        # Create DuckDB with episode stats schema
        conn, db_path = create_episode_stats_db()

        episode_id = str(uuid.uuid4())
        pv_id_str = str(pv_id)

        insert_episode(
            conn,
            episode_id=episode_id,
            primary_pv_id=pv_id_str,
            replay_url=None,
            thumbnail_url=None,
            attributes={},
            eval_task_id=None,
        )

        # Add agent with multiple metrics
        insert_agent_policy(conn, episode_id, pv_id_str, 0)
        insert_agent_metric(conn, episode_id, 0, "reward", 5.0)
        insert_agent_metric(conn, episode_id, 0, "steps", 100.0)
        insert_agent_metric(conn, episode_id, 0, "custom_metric", 42.0)

        conn.close()

        self._setup_s3_mocks(mock_aioboto3, db_path)
        self._call_presigned_upload_flow(test_client, auth_headers)

        # Verify only reward metric is stored
        async with stats_repo.connect() as con:
            result = await con.execute(
                """
                SELECT DISTINCT metric_name
                FROM episode_policy_metrics
                """
            )
            metrics = await result.fetchall()
            metric_names = [m[0] for m in metrics]

            # Only 'reward' should be present
            assert "reward" in metric_names
            assert "steps" not in metric_names
            assert "custom_metric" not in metric_names
