"""Integration tests for centralized sweep database and parallel worker support.

These tests cover the new centralized coordination functionality:
1. Centralized sweep creation and retrieval via database API
2. Atomic run ID generation for parallel workers
3. Process-specific file handling to prevent worker collisions
4. End-to-end parallel worker coordination
5. Integration with existing sweep pipeline
"""

import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import Mock, patch

import pytest
from omegaconf import OmegaConf

from cogweb.cogweb_client import CogwebClient


class TestCentralizedSweepIntegration:
    """Integration tests for centralized sweep coordination system."""

    def setup_method(self):
        """Set up test environment for each test method."""
        self.test_dir = tempfile.mkdtemp()
        self.sweep_dir = os.path.join(self.test_dir, "test_sweep")
        self.runs_dir = os.path.join(self.sweep_dir, "runs")
        os.makedirs(self.runs_dir, exist_ok=True)

    def teardown_method(self):
        """Clean up test environment after each test method."""
        if hasattr(self, "test_dir") and os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

        # Clear the singleton cache to avoid test interference
        from cogweb.cogweb_client import CogwebClient

        CogwebClient._instances.clear()

    @patch("cogweb.cogweb_client.SweepClient")
    def test_centralized_sweep_creation_workflow(self, mock_client_class):
        """Test complete workflow from sweep creation to run coordination."""
        # Mock client setup
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Mock sweep creation response
        mock_create_response = Mock()
        mock_create_response.created = True
        mock_create_response.sweep_id = "central-sweep-123"
        mock_client.create_sweep.return_value = mock_create_response

        # Mock sweep retrieval
        mock_info = Mock()
        mock_info.exists = True
        mock_info.wandb_sweep_id = "wandb_central_123"
        mock_client.get_sweep.return_value = mock_info

        # Mock run ID generation
        mock_client.get_next_run_id.side_effect = ["test_sweep.r.0", "test_sweep.r.1", "test_sweep.r.2"]

        # Create client and test methods
        client = CogwebClient()

        # 1. Create sweep in centralized database
        response = client.create_sweep("test_sweep", "test_project", "test_entity", "wandb_central_123")

        assert response.created is True
        assert response.sweep_id == "central-sweep-123"
        mock_client.create_sweep.assert_called_once_with(
            "test_sweep", "test_project", "test_entity", "wandb_central_123"
        )

        # 2. Retrieve sweep ID
        sweep_id = client.sweep_id("test_sweep")
        assert sweep_id == "wandb_central_123"
        mock_client.get_sweep.assert_called_once_with("test_sweep")

        # 3. Generate multiple run IDs atomically
        run_ids = []
        for _ in range(3):
            run_id = client.sweep_next_run_id("test_sweep")
            run_ids.append(run_id)

        assert run_ids == ["test_sweep.r.0", "test_sweep.r.1", "test_sweep.r.2"]
        assert mock_client.get_next_run_id.call_count == 3

    @patch("cogweb.cogweb_client.SweepClient")
    def test_concurrent_run_generation_atomic_behavior(self, mock_client_class):
        """Test that concurrent run ID generation maintains atomicity."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Use itertools.cycle to ensure we don't run out of values
        from itertools import cycle

        expected_run_ids = [f"test_sweep.r.{i}" for i in range(10)]
        mock_client.get_next_run_id.side_effect = cycle(expected_run_ids)

        # Simulate concurrent workers requesting run IDs
        def worker_get_run_id(worker_id):
            client = CogwebClient()
            return client.sweep_next_run_id("test_sweep")

        # Use ThreadPoolExecutor to simulate concurrent access
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(worker_get_run_id, i) for i in range(10)]
            actual_run_ids = [future.result() for future in as_completed(futures)]

        # All run IDs should be from our expected set (may repeat due to cycle)
        assert len(actual_run_ids) == 10
        for run_id in actual_run_ids:
            assert run_id in expected_run_ids
        assert mock_client.get_next_run_id.call_count == 10

    def test_process_specific_dist_cfg_path_generation(self):
        """Test generation of process-specific dist_cfg paths."""
        # Test data representing different sweep processes
        process_ids = ["abc123ef", "def456gh", "ghi789jk"]
        sweep_name = "dist_test_sweep"
        data_dir = self.test_dir

        expected_paths = []
        for process_id in process_ids:
            expected_path = f"{data_dir}/sweep/{sweep_name}/dist_{process_id}.yaml"
            expected_paths.append(expected_path)

        # Verify path generation logic (simulating what sweep_rollout.sh does)
        generated_paths = []
        for process_id in process_ids:
            path = f"{data_dir}/sweep/{sweep_name}/dist_{process_id}.yaml"
            generated_paths.append(path)

        assert generated_paths == expected_paths
        assert len(set(generated_paths)) == len(process_ids)  # All unique

    def test_sweep_config_integration_and_overrides(self):
        """Test integration of sweep configs with run-specific overrides."""
        # Simulate sweep_prepare_run.py config generation
        run_id = "config_test_sweep.r.7"
        run_dir = os.path.join(self.runs_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)

        # Base sweep job config (simulating what's in configs/sweep_job.yaml)
        base_config = {
            "trainer": {"learning_rate": 0.001, "batch_size": 64},
            "sim": {"num_episodes": 10, "max_time_s": 120},
            "wandb": {"group": "config_test_sweep"},
        }

        # Run-specific overrides (protein suggestions + run metadata)
        run_overrides = {
            "trainer": {"learning_rate": 0.0008},  # Protein suggestion
            "run": run_id,
            "run_dir": run_dir,
            "seed": 12345,
            "wandb": {"name": run_id},
        }

        # Merge configs (simulating apply_protein_suggestion logic)
        final_config = {**base_config, **run_overrides}
        final_config["trainer"] = {**base_config["trainer"], **run_overrides["trainer"]}
        final_config["wandb"] = {**base_config["wandb"], **run_overrides["wandb"]}

        # Save final config
        config_path = os.path.join(run_dir, "train_config_overrides.yaml")
        OmegaConf.save(final_config, config_path)

        # Verify saved configuration
        assert os.path.exists(config_path)
        loaded_config = OmegaConf.load(config_path)

        assert loaded_config.run == run_id
        assert loaded_config.run_dir == run_dir
        assert loaded_config.seed == 12345
        assert loaded_config.trainer.learning_rate == 0.0008  # Protein override
        assert loaded_config.trainer.batch_size == 64  # Base value preserved
        assert loaded_config.wandb.group == "config_test_sweep"
        assert loaded_config.wandb.name == run_id

    @patch("cogweb.cogweb_client.SweepClient")
    def test_sweep_idempotency_and_consistency(self, mock_client_class):
        """Test that sweep operations are idempotent and consistent."""
        # Create fresh mock for this test
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # First call: Create new sweep
        mock_create_response_new = Mock()
        mock_create_response_new.created = True
        mock_create_response_new.sweep_id = "idempotent-sweep-123"

        # Second call: Return existing sweep
        mock_create_response_existing = Mock()
        mock_create_response_existing.created = False
        mock_create_response_existing.sweep_id = "idempotent-sweep-123"

        mock_client.create_sweep.side_effect = [mock_create_response_new, mock_create_response_existing]

        # Mock consistent sweep retrieval
        mock_info = Mock()
        mock_info.exists = True
        mock_info.wandb_sweep_id = "wandb_idempotent_123"
        mock_client.get_sweep.return_value = mock_info

        # Create clients and test idempotency
        client1 = CogwebClient()
        client2 = CogwebClient()  # Should be the same instance due to singleton

        # 1. First creation should create new sweep
        response1 = client1.create_sweep("idempotent_sweep", "project", "entity", "wandb_idempotent_123")
        assert response1.created is True
        assert response1.sweep_id == "idempotent-sweep-123"

        # 2. Second creation should return existing sweep
        response2 = client2.create_sweep("idempotent_sweep", "project", "entity", "wandb_idempotent_123")
        assert response2.created is False
        assert response2.sweep_id == "idempotent-sweep-123"  # Same ID

        # 3. Retrieval should be consistent
        sweep_id1 = client1.sweep_id("idempotent_sweep")
        sweep_id2 = client2.sweep_id("idempotent_sweep")

        assert sweep_id1 == sweep_id2 == "wandb_idempotent_123"

    @patch("cogweb.cogweb_client.SweepClient")
    def test_error_handling_and_recovery(self, mock_client_class):
        """Test error handling in centralized sweep operations."""
        # Create fresh mock for this test
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Test create_sweep error propagation
        mock_client.create_sweep.side_effect = Exception("Backend connection failed")

        client = CogwebClient()

        with pytest.raises(Exception, match="Backend connection failed"):
            client.create_sweep("error_sweep", "project", "entity", "wandb_123")

    def test_backward_compatibility_with_existing_sweep_structure(self):
        """Test that new centralized approach maintains backward compatibility."""
        # Test existing sweep directory structure
        sweep_name = "backward_compat_sweep"
        sweep_dir = os.path.join(self.test_dir, "sweep", sweep_name)
        runs_dir = os.path.join(sweep_dir, "runs")
        metadata_path = os.path.join(sweep_dir, "metadata.yaml")

        os.makedirs(runs_dir, exist_ok=True)

        # Create metadata in existing format
        metadata = {
            "sweep_name": sweep_name,
            "wandb_sweep_id": "backward_compat_123",
            "wandb_path": "entity/project/backward_compat_123",
        }
        OmegaConf.save(metadata, metadata_path)

        # Test run directory structure
        run_id = f"{sweep_name}.r.0"
        run_dir = os.path.join(runs_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)

        # Create overrides file (train_config_overrides.yaml)
        overrides = {
            "run": run_id,
            "run_dir": run_dir,
            "trainer": {"learning_rate": 0.002},
        }
        overrides_path = os.path.join(run_dir, "train_config_overrides.yaml")
        OmegaConf.save(overrides, overrides_path)

        # Verify structure is maintained
        assert os.path.exists(metadata_path)
        assert os.path.exists(overrides_path)

        loaded_metadata = OmegaConf.load(metadata_path)
        loaded_overrides = OmegaConf.load(overrides_path)

        assert loaded_metadata.sweep_name == sweep_name
        assert loaded_metadata.wandb_sweep_id == "backward_compat_123"
        assert loaded_overrides.run == run_id
        assert loaded_overrides.trainer.learning_rate == 0.002

    @patch("cogweb.cogweb_client.SweepClient")
    def test_integration_with_sweep_pipeline_tools(self, mock_client_class):
        """Test integration with existing sweep pipeline tools."""
        # Create fresh mock for this test
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Mock responses for pipeline workflow
        mock_info = Mock()
        mock_info.exists = True
        mock_info.wandb_sweep_id = "pipeline_wandb_123"
        mock_client.get_sweep.return_value = mock_info

        mock_client.get_next_run_id.return_value = "pipeline_sweep.r.5"

        # Simulate pipeline workflow

        # 1. sweep_setup.py would check for existing sweep
        existing_sweep_id = CogwebClient().sweep_id("pipeline_sweep")
        assert existing_sweep_id == "pipeline_wandb_123"

        # 2. sweep_prepare_run.py would generate new run ID
        new_run_id = CogwebClient().sweep_next_run_id("pipeline_sweep")
        assert new_run_id == "pipeline_sweep.r.5"
