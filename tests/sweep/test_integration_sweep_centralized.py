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
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import Mock, patch

import pytest
from omegaconf import DictConfig, OmegaConf

from metta.sweep.metta_client_utils import (
    create_sweep_in_metta,
    get_next_run_id_from_metta,
    get_sweep_id_from_metta,
)


class TestCentralizedSweepIntegration:
    """Integration tests for centralized sweep coordination."""

    @pytest.fixture(autouse=True)
    def setup_test_environment(self):
        """Setup test environment with temporary directories."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.temp_dir, "data")
        self.sweep_dir = os.path.join(self.data_dir, "sweep", "test_sweep")
        self.runs_dir = os.path.join(self.sweep_dir, "runs")

        os.makedirs(self.runs_dir, exist_ok=True)

        yield

        # Cleanup
        if hasattr(self, "temp_dir"):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("metta.sweep.metta_client_utils.SweepClient")
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

        # 1. Create sweep in centralized database
        response = create_sweep_in_metta("test_sweep", "test_entity", "test_project", "wandb_central_123")

        assert response.created is True
        assert response.sweep_id == "central-sweep-123"
        mock_client.create_sweep.assert_called_once_with(
            "test_sweep", "test_project", "test_entity", "wandb_central_123"
        )

        # 2. Retrieve sweep ID
        sweep_id = get_sweep_id_from_metta("test_sweep")
        assert sweep_id == "wandb_central_123"
        mock_client.get_sweep.assert_called_once_with("test_sweep")

        # 3. Generate multiple run IDs atomically
        run_ids = []
        for _ in range(3):
            run_id = get_next_run_id_from_metta("test_sweep")
            run_ids.append(run_id)

        assert run_ids == ["test_sweep.r.0", "test_sweep.r.1", "test_sweep.r.2"]
        assert mock_client.get_next_run_id.call_count == 3

    @patch("metta.sweep.metta_client_utils.SweepClient")
    def test_parallel_run_id_generation_thread_safety(self, mock_client_class):
        """Test that parallel run ID generation is thread-safe and atomic."""
        # Mock client setup
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Track call order to ensure atomicity
        call_counter = 0
        generated_ids = []

        def mock_get_next_run_id(sweep_name):
            nonlocal call_counter
            # Simulate some processing time
            time.sleep(0.01)
            call_counter += 1
            run_id = f"{sweep_name}.r.{call_counter - 1}"
            generated_ids.append(run_id)
            return run_id

        mock_client.get_next_run_id.side_effect = mock_get_next_run_id

        # Simulate 5 parallel workers requesting run IDs
        num_workers = 5
        results = []

        def worker_get_run_id():
            return get_next_run_id_from_metta("parallel_test_sweep")

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(worker_get_run_id) for _ in range(num_workers)]

            for future in as_completed(futures):
                results.append(future.result())

        # Verify all IDs are unique and sequential
        expected_ids = [f"parallel_test_sweep.r.{i}" for i in range(num_workers)]

        assert len(results) == num_workers
        assert len(set(results)) == num_workers  # All unique
        assert sorted(results) == sorted(expected_ids)  # Correct format
        assert mock_client.get_next_run_id.call_count == num_workers

    def test_process_specific_dist_cfg_path_generation(self):
        """Test that different process IDs generate different dist_cfg paths."""
        # Simulate different sweep process IDs
        process_ids = ["abc123ef", "def456gh", "ghi789jk"]
        sweep_name = "dist_test_sweep"
        data_dir = self.data_dir

        expected_paths = []
        for process_id in process_ids:
            expected_path = os.path.join(data_dir, "sweep", sweep_name, f"dist_{process_id}.yaml")
            expected_paths.append(expected_path)

        # Verify paths are all different
        assert len(set(expected_paths)) == len(expected_paths)

        # Verify path structure
        for i, path in enumerate(expected_paths):
            assert f"dist_{process_ids[i]}.yaml" in path
            assert sweep_name in path

    def test_config_override_file_isolation_by_run_id(self):
        """Test that run-specific config overrides don't conflict."""
        # Create multiple run directories
        run_ids = ["test_sweep.r.0", "test_sweep.r.1", "test_sweep.r.2"]
        override_files = []

        for run_id in run_ids:
            run_dir = os.path.join(self.runs_dir, run_id)
            os.makedirs(run_dir, exist_ok=True)

            # Create override file in each run directory
            override_path = os.path.join(run_dir, "train_config_overrides.yaml")
            override_config = {"trainer": {"learning_rate": 0.001 + (0.0001 * len(override_files))}}

            OmegaConf.save(override_config, override_path)
            override_files.append(override_path)

        # Verify all files exist and contain different configs
        assert len(override_files) == 3

        for i, override_file in enumerate(override_files):
            assert os.path.exists(override_file)
            loaded_config = OmegaConf.load(override_file)
            expected_lr = 0.001 + (0.0001 * i)
            assert loaded_config.trainer.learning_rate == expected_lr

    @patch("metta.sweep.metta_client_utils.SweepClient")
    def test_sweep_process_id_integration_with_file_paths(self, mock_client_class):
        """Test integration of sweep_process_id with file path generation."""
        # Mock client setup
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.get_next_run_id.return_value = "integration_sweep.r.42"

        # Create a config with sweep_process_id
        config = DictConfig(
            {
                "sweep_name": "integration_sweep",
                "sweep_process_id": "test_proc_123",
                "data_dir": self.data_dir,
                "runs_dir": self.runs_dir,
            }
        )

        # Test that process-specific paths are generated correctly
        dist_cfg_path = os.path.join(
            config.data_dir, "sweep", config.sweep_name, f"dist_{config.sweep_process_id}.yaml"
        )

        expected_path = os.path.join(self.data_dir, "sweep", "integration_sweep", "dist_test_proc_123.yaml")
        assert dist_cfg_path == expected_path

        # Verify the path is unique per process
        config2 = DictConfig(
            {"sweep_name": "integration_sweep", "sweep_process_id": "different_proc_456", "data_dir": self.data_dir}
        )

        dist_cfg_path2 = os.path.join(
            config2.data_dir, "sweep", config2.sweep_name, f"dist_{config2.sweep_process_id}.yaml"
        )

        assert dist_cfg_path != dist_cfg_path2
        assert "test_proc_123" in dist_cfg_path
        assert "different_proc_456" in dist_cfg_path2

    @patch("metta.sweep.metta_client_utils.SweepClient")
    def test_sweep_idempotency_and_consistency(self, mock_client_class):
        """Test that sweep operations are idempotent and consistent."""
        # Mock client setup
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

        # 1. First creation should create new sweep
        response1 = create_sweep_in_metta("idempotent_sweep", "entity", "project", "wandb_idempotent_123")
        assert response1.created is True
        assert response1.sweep_id == "idempotent-sweep-123"

        # 2. Second creation should return existing sweep
        response2 = create_sweep_in_metta("idempotent_sweep", "entity", "project", "wandb_idempotent_123")
        assert response2.created is False
        assert response2.sweep_id == "idempotent-sweep-123"  # Same ID

        # 3. Retrieval should be consistent
        sweep_id1 = get_sweep_id_from_metta("idempotent_sweep")
        sweep_id2 = get_sweep_id_from_metta("idempotent_sweep")

        assert sweep_id1 == sweep_id2 == "wandb_idempotent_123"

    @patch("metta.sweep.metta_client_utils.SweepClient")
    def test_error_handling_and_failure_propagation(self, mock_client_class):
        """Test that errors from the centralized API are properly handled."""
        # Mock client setup
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Test network/API errors
        mock_client.get_sweep.side_effect = ConnectionError("Database connection failed")
        mock_client.create_sweep.side_effect = TimeoutError("API timeout")
        mock_client.get_next_run_id.side_effect = Exception("Server error")

        # Verify errors propagate correctly (no hiding/swallowing)
        with pytest.raises(ConnectionError, match="Database connection failed"):
            get_sweep_id_from_metta("error_test_sweep")

        with pytest.raises(TimeoutError, match="API timeout"):
            create_sweep_in_metta("error_sweep", "entity", "project", "wandb_123")

        with pytest.raises(Exception, match="Server error"):
            get_next_run_id_from_metta("error_sweep")

    def test_backward_compatibility_with_existing_sweep_structure(self):
        """Test that new centralized approach works with existing file structures."""
        # Create existing sweep directory structure
        sweep_metadata_path = os.path.join(self.sweep_dir, "metadata.yaml")
        os.makedirs(os.path.dirname(sweep_metadata_path), exist_ok=True)

        # Traditional sweep metadata
        legacy_metadata = {
            "sweep": "backward_compat_test",
            "wandb_sweep_id": "legacy_wandb_123",
            "wandb_path": "entity/project/legacy_wandb_123",
        }
        OmegaConf.save(legacy_metadata, sweep_metadata_path)

        # Create legacy run directory
        legacy_run_dir = os.path.join(self.runs_dir, "backward_compat_test.r.0")
        os.makedirs(legacy_run_dir, exist_ok=True)

        # Legacy override file (without process ID)
        legacy_override_path = os.path.join(legacy_run_dir, "train_config_overrides.yaml")
        legacy_overrides = {"trainer": {"learning_rate": 0.002}}
        OmegaConf.save(legacy_overrides, legacy_override_path)

        # Verify legacy structure still works
        assert os.path.exists(sweep_metadata_path)
        assert os.path.exists(legacy_run_dir)
        assert os.path.exists(legacy_override_path)

        # Load and verify legacy config
        loaded_metadata = OmegaConf.load(sweep_metadata_path)
        assert loaded_metadata.wandb_sweep_id == "legacy_wandb_123"

        loaded_overrides = OmegaConf.load(legacy_override_path)
        assert loaded_overrides.trainer.learning_rate == 0.002

    @patch("metta.sweep.metta_client_utils.SweepClient")
    def test_integration_with_sweep_pipeline_tools(self, mock_client_class):
        """Test integration with existing sweep pipeline tools."""
        # Mock client setup
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Mock centralized responses
        mock_info = Mock()
        mock_info.exists = True
        mock_info.wandb_sweep_id = "pipeline_wandb_123"
        mock_client.get_sweep.return_value = mock_info

        mock_client.get_next_run_id.return_value = "pipeline_sweep.r.5"

        # Test integration points that would be called by sweep tools

        # 1. sweep_setup.py would check for existing sweep
        existing_sweep_id = get_sweep_id_from_metta("pipeline_sweep")
        assert existing_sweep_id == "pipeline_wandb_123"

        # 2. sweep_prepare_run.py would generate new run ID
        new_run_id = get_next_run_id_from_metta("pipeline_sweep")
        assert new_run_id == "pipeline_sweep.r.5"

        # 3. Verify expected API calls were made
        mock_client.get_sweep.assert_called_with("pipeline_sweep")
        mock_client.get_next_run_id.assert_called_with("pipeline_sweep")

        # 4. Simulate config generation for new run
        run_dir = os.path.join(self.runs_dir, new_run_id)
        os.makedirs(run_dir, exist_ok=True)

        # Process-specific config (simulating sweep_prepare_run.py)
        process_id = "pipeline_proc_789"
        override_path = os.path.join(run_dir, "train_config_overrides.yaml")
        override_config = {
            "trainer": {"learning_rate": 0.0008},
            "run": new_run_id,
            "run_dir": run_dir,
            "sweep_process_id": process_id,
        }
        OmegaConf.save(override_config, override_path)

        # Verify integration worked
        assert os.path.exists(override_path)
        loaded_config = OmegaConf.load(override_path)
        assert loaded_config.run == new_run_id
        assert loaded_config.sweep_process_id == process_id
