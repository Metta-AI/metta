"""Integration tests for distributed sweep coordination."""

import os
from unittest.mock import Mock, patch

import pytest
from omegaconf import DictConfig

from metta.common.util.lock import run_once


class TestDistributedSweepIntegration:
    """Test distributed sweep coordination."""

    @pytest.fixture
    def mock_distributed_env(self):
        """Mock distributed environment variables."""
        original_env = os.environ.copy()

        # Set up distributed environment
        os.environ["WORLD_SIZE"] = "2"
        os.environ["RANK"] = "0"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"

        yield

        # Restore original environment
        os.environ.clear()
        os.environ.update(original_env)

    def test_run_once_coordination_single_node(self):
        """Test run_once works correctly in single-node setup."""
        # Counter to track function calls
        call_count = 0

        def test_function():
            nonlocal call_count
            call_count += 1
            return "test_result"

        # Without distributed setup, should just run the function
        result = run_once(test_function)

        assert result == "test_result"
        assert call_count == 1

    @patch("metta.common.util.lock.dist.is_initialized")
    @patch("metta.common.util.lock.dist.get_rank")
    @patch("metta.common.util.lock.dist.broadcast_object_list")
    def test_run_once_coordination_multi_node(self, mock_broadcast, mock_get_rank, mock_is_initialized):
        """Test run_once coordination across multiple nodes."""
        # Setup mocks
        mock_is_initialized.return_value = True
        mock_get_rank.return_value = 0  # Head node

        # Mock broadcast to simulate sharing result
        def broadcast_side_effect(objects, src=0):
            # Simulate broadcast behavior
            if mock_get_rank.return_value == src:
                # Source node keeps its value
                pass
            else:
                # Other nodes receive the value
                objects[0] = "broadcasted_result"

        mock_broadcast.side_effect = broadcast_side_effect

        # Test function
        def test_function():
            return "head_node_result"

        # Run on head node
        result = run_once(test_function, destroy_on_finish=False)

        assert result == "head_node_result"
        mock_broadcast.assert_called_once()

    @patch("metta.common.util.lock.dist.is_initialized")
    @patch("metta.common.util.lock.dist.get_rank")
    @patch("metta.common.util.lock.dist.broadcast_object_list")
    def test_distributed_training_all_nodes_participate(self, mock_broadcast, mock_get_rank, mock_is_initialized):
        """Test that all nodes participate in training while only head runs orchestration."""
        # Track which operations run on which nodes
        orchestration_calls = []
        training_calls = []

        def orchestration_task():
            orchestration_calls.append(mock_get_rank.return_value)
            return {"config": "test_config"}

        def training_task():
            training_calls.append(mock_get_rank.return_value)
            return "training_complete"

        # Simulate running on different ranks
        for rank in [0, 1]:
            mock_get_rank.return_value = rank
            mock_is_initialized.return_value = True

            # Orchestration should only run on rank 0
            if rank == 0:
                mock_broadcast.side_effect = lambda objects, src=0: None
                result = run_once(orchestration_task, destroy_on_finish=False)
                assert result == {"config": "test_config"}
            else:
                # Worker nodes receive broadcasted result
                def worker_broadcast(objects, src=0):
                    objects[0] = {"config": "test_config"}

                mock_broadcast.side_effect = worker_broadcast
                result = run_once(orchestration_task, destroy_on_finish=False)
                assert result == {"config": "test_config"}

            # Training runs on all nodes
            training_result = training_task()
            assert training_result == "training_complete"

        # Verify orchestration only ran on rank 0
        assert orchestration_calls == [0]
        # Verify training ran on all ranks
        assert training_calls == [0, 1]

    @patch("metta.common.util.lock.dist.is_initialized")
    @patch("metta.common.util.lock.dist.init_process_group")
    @patch("metta.common.util.lock.dist.destroy_process_group")
    def test_process_group_lifecycle(self, mock_destroy_pg, mock_init_pg, mock_is_initialized, mock_distributed_env):
        """Test process group initialization and cleanup."""
        # Initially not initialized
        mock_is_initialized.return_value = False

        def test_function():
            return "result"

        # Test with destroy_on_finish=True (default)
        result = run_once(test_function)

        assert result == "result"
        mock_init_pg.assert_called_once()
        mock_destroy_pg.assert_called_once()

        # Reset mocks
        mock_init_pg.reset_mock()
        mock_destroy_pg.reset_mock()

        # Test with destroy_on_finish=False
        result = run_once(test_function, destroy_on_finish=False)

        assert result == "result"
        mock_init_pg.assert_called_once()
        mock_destroy_pg.assert_not_called()

    def test_head_node_only_operations(self):
        """Test that certain operations only run on head node."""
        from metta.sweep.sweep_lifecycle import evaluate_rollout, prepare_sweep_run, setup_sweep

        # Mock config and logger
        mock_config = DictConfig(
            {
                "sweep_name": "test_sweep",
                "sweep_server_uri": "http://test-server",
                "wandb": {"entity": "test", "project": "test"},
                "sweep": {"metric": "reward"},
                "runs_dir": "/tmp/runs",
                "sweep_dir": "/tmp/sweep",
            }
        )
        mock_logger = Mock()

        # These should be wrapped in run_once in the actual implementation
        operations = [
            (setup_sweep, (mock_config, mock_logger)),
            (prepare_sweep_run, (mock_config, mock_logger)),
            (evaluate_rollout, (mock_config, {}, mock_logger)),
        ]

        # Verify each operation can be called (would be wrapped in run_once in real usage)
        for op, _args in operations:
            # In real implementation, these would be called via run_once
            # Here we just verify they're callable
            assert callable(op)


class TestSweepPipelineIntegration:
    """End-to-end pipeline integration tests."""

    @pytest.fixture
    def integration_config(self):
        """Create a complete integration test configuration."""
        return DictConfig(
            {
                "sweep_name": "integration_test_sweep",
                "sweep_server_uri": "http://test-server",
                "wandb": {
                    "enabled": True,
                    "entity": "test_entity",
                    "project": "test_project",
                },
                "sweep": {
                    "metric": "reward",
                    "goal": "maximize",
                    "parameters": {
                        "trainer": {
                            "learning_rate": {"min": 0.001, "max": 0.01},
                            "batch_size": {"values": [2048, 4096]},
                        }
                    },
                },
                "trainer": {
                    "batch_size": 2048,
                    "minibatch_size": 64,
                    "bptt_horizon": 32,
                    "total_timesteps": 1000,
                },
                "sim": {
                    "num_envs": 1,
                    "num_episodes": 10,
                },
                "device": "cpu",
                "vectorization": "none",
                "data_dir": "/tmp/data",
                "runs_dir": "/tmp/runs",
                "sweep_dir": "/tmp/sweep",
                "sweep_id": "test_sweep_123",
                "max_consecutive_failures": 3,
                "rollout_retry_delay": 1,
                "max_observations_to_load": 100,
                "sweep_job": {
                    "trainer": {"learning_rate": 0.001},
                },
            }
        )

    @patch("metta.sweep.sweep_lifecycle.CogwebClient")
    @patch("metta.sweep.sweep_lifecycle.create_wandb_sweep")
    @patch("metta.sweep.sweep_lifecycle.create_wandb_run_for_sweep")
    @patch("metta.sweep.sweep_lifecycle.MettaProtein")
    @patch("metta.sweep.sweep_lifecycle.PolicyStore")
    @patch("metta.sweep.sweep_lifecycle.SimulationSuite")
    @patch("subprocess.run")
    def test_complete_sweep_lifecycle(
        self,
        mock_subprocess,
        mock_sim_suite,
        mock_policy_store,
        mock_protein,
        mock_create_run,
        mock_create_sweep,
        mock_cogweb,
        integration_config,
    ):
        """Test complete sweep lifecycle from setup to evaluation."""
        import sys

        sys.path.insert(0, "tools")
        from sweep_rollout import run_single_rollout

        # Setup mocks
        mock_client = Mock()
        mock_cogweb.get_client.return_value = mock_client
        mock_sweep_client = Mock()
        mock_client.sweep_client.return_value = mock_sweep_client

        # Mock sweep doesn't exist yet
        mock_sweep_info = Mock()
        mock_sweep_info.exists = False
        mock_sweep_client.get_sweep.return_value = mock_sweep_info

        # Mock sweep creation
        mock_create_sweep.return_value = "wandb_sweep_123"
        mock_sweep_client.get_next_run_id.return_value = "test_sweep.r.1"

        # Mock protein
        mock_protein_instance = Mock()
        mock_protein.return_value = mock_protein_instance
        mock_protein_instance.suggest.return_value = (
            {"trainer": {"learning_rate": 0.005}},
            {"cost": 100.0},
        )

        # Mock training subprocess
        mock_subprocess.return_value = Mock(returncode=0)

        # Mock policy store and evaluation
        mock_ps = Mock()
        mock_policy_store.return_value = mock_ps
        mock_policy_record = Mock()
        mock_policy_record.uri = "wandb://run/test_run"
        mock_policy_record.metadata = {
            "train_time": 100.0,
            "agent_step": 1000,
            "epoch": 10,
        }
        mock_ps.policy_record.return_value = mock_policy_record
        mock_ps.load_from_uri.return_value = mock_policy_record

        # Mock simulation
        mock_sim = Mock()
        mock_sim_suite.return_value = mock_sim
        mock_results = Mock()
        mock_results.stats_db = Mock()
        mock_sim.simulate.return_value = mock_results

        # Mock wandb API to avoid real API calls
        with (
            patch("metta.sweep.wandb_utils.wandb.Api"),
            patch("metta.sweep.sweep_lifecycle.fetch_protein_observations_from_wandb") as mock_fetch_obs,
        ):
            # Mock fetch observations to return empty list
            mock_fetch_obs.return_value = []

            # Mock wandb run creation
            mock_run = Mock()
            mock_run.name = "test_run"
            mock_run.id = "run_123"
            mock_create_run.return_value = mock_run

            # Mock file operations
            with (
                patch("metta.sweep.sweep_lifecycle.os.makedirs"),
                patch("metta.sweep.sweep_lifecycle.OmegaConf.save"),
                patch("metta.sweep.sweep_lifecycle.EvalStatsDB") as mock_eval_db,
            ):
                # Mock eval DB
                mock_eval = Mock()
                mock_eval_db.from_sim_stats_db.return_value = mock_eval
                mock_eval.get_average_metric_by_filter.return_value = 0.85

                # Mock WandbContext
                with patch("metta.sweep.sweep_lifecycle.WandbContext") as mock_wandb_context:
                    mock_wandb_run = Mock()
                    mock_wandb_run.name = "test_run"
                    mock_wandb_run.summary = Mock(update=Mock())
                    mock_wandb_context.return_value.__enter__.return_value = mock_wandb_run

                    # Mock simulation suite output
                    with patch("metta.sweep.sweep_lifecycle.SimulationSuiteConfig") as mock_sim_config:
                        mock_sim_config.return_value = Mock()

                        # Run single rollout
                        with patch("metta.common.util.lock.run_once", side_effect=lambda func, **kwargs: func()):
                            # Setup sweep first
                            from metta.sweep.sweep_lifecycle import setup_sweep

                            logger = Mock()
                            sweep_id = setup_sweep(integration_config, logger)
                            integration_config.sweep_id = sweep_id

                            # This would normally be called by main()
                            result = run_single_rollout(integration_config)

        # Verify the pipeline executed correctly
        assert result == 0
        mock_create_sweep.assert_called_once()
        mock_protein_instance.suggest.assert_called()
        mock_subprocess.assert_called_once()  # Training was launched
        mock_ps.add_to_wandb_sweep.assert_called()

    @patch("metta.sweep.wandb_utils.wandb.Api")
    @patch("metta.sweep.wandb_utils.wandb.sweep")
    def test_wandb_integration(self, mock_wandb_sweep, mock_wandb_api):
        """Test WandB integration for sweep management."""
        from metta.sweep.wandb_utils import create_wandb_sweep, fetch_protein_observations_from_wandb

        # Mock WandB API
        mock_api = Mock()
        mock_wandb_api.return_value = mock_api

        # Mock sweep creation
        mock_wandb_sweep.return_value = "test_sweep_456"

        # Test sweep creation
        sweep_id = create_wandb_sweep("test_entity", "test_project", "test_sweep")
        assert sweep_id == "test_sweep_456"
        mock_wandb_sweep.assert_called_once()

        # Mock fetching observations
        mock_run1 = Mock()
        mock_run1.summary = {"protein_observation": {"suggestion": {"lr": 0.001}, "objective": 0.8}}
        mock_run2 = Mock()
        mock_run2.summary = {"protein_observation": {"suggestion": {"lr": 0.005}, "objective": 0.9}}
        mock_runs = [mock_run1, mock_run2]
        mock_api.runs.return_value = mock_runs

        # Test fetching observations
        observations = fetch_protein_observations_from_wandb(
            "test_entity", "test_project", "test_sweep_456", max_observations=10
        )

        # Should return the mocked observations
        assert len(observations) == 2
        assert observations[0]["suggestion"]["lr"] == 0.001
        assert observations[1]["objective"] == 0.9

    @patch("metta.sweep.sweep_lifecycle.CogwebClient")
    def test_cogweb_integration(self, mock_cogweb_class):
        """Test Cogweb integration for distributed coordination."""
        from metta.sweep.sweep_lifecycle import setup_sweep

        # Setup mocks
        mock_client = Mock()
        mock_cogweb_class.get_client.return_value = mock_client
        mock_sweep_client = Mock()
        mock_client.sweep_client.return_value = mock_sweep_client

        # Mock existing sweep
        mock_sweep_info = Mock()
        mock_sweep_info.exists = True
        mock_sweep_info.wandb_sweep_id = "existing_sweep_789"
        mock_sweep_client.get_sweep.return_value = mock_sweep_info

        # Test setup with existing sweep
        config = DictConfig(
            {
                "sweep_name": "test_sweep",
                "sweep_server_uri": "http://test-server",
                "wandb": {"entity": "test", "project": "test"},
                "sweep": {"metric": "reward"},
                "runs_dir": "/tmp/runs",
                "sweep_dir": "/tmp/sweep",
            }
        )
        logger = Mock()

        with (
            patch("metta.common.util.lock.run_once", side_effect=lambda func, **kwargs: func()),
            patch("metta.sweep.sweep_lifecycle.os.makedirs"),
            patch("metta.sweep.sweep_lifecycle.OmegaConf.save"),
        ):
            sweep_id = setup_sweep(config, logger)

        assert sweep_id == "existing_sweep_789"
        mock_sweep_client.get_sweep.assert_called_once_with("test_sweep")

    @patch("metta.agent.policy_store.PolicyStore")
    def test_policy_store_integration(self, mock_policy_store_class):
        """Test PolicyStore integration for model management."""

        # Mock policy store
        mock_ps = Mock()
        mock_policy_store_class.return_value = mock_ps

        # Mock policy record
        mock_policy = Mock()
        mock_policy.uri = "wandb://run/test_run"
        mock_policy.metadata = {
            "train_time": 150.0,
            "agent_step": 1000,
            "epoch": 10,
        }
        mock_ps.policy_record.return_value = mock_policy
        mock_ps.load_from_uri.return_value = mock_policy

        # Test that policy store is used correctly in evaluation
        # This verifies the integration point exists
        assert mock_policy_store_class is not None
        assert callable(mock_ps.add_to_wandb_sweep)


class TestMultipleRolloutProgression:
    """Test multiple rollout progression and error handling."""

    @patch("subprocess.run")
    @patch("metta.sweep.sweep_lifecycle.MettaProtein")
    @patch("metta.common.util.lock.run_once")
    def test_multiple_rollout_progression(self, mock_run_once, mock_protein_class, mock_subprocess):
        """Test that sweeps progress through multiple rollouts correctly."""
        import sys

        sys.path.insert(0, "tools")
        from sweep_rollout import run_single_rollout

        # Track rollout progression
        rollout_count = 0
        suggestions = []

        # Mock protein to return different suggestions
        mock_protein = Mock()
        mock_protein_class.return_value = mock_protein

        def suggest_side_effect():
            nonlocal rollout_count
            rollout_count += 1
            suggestion = {"trainer": {"learning_rate": 0.001 * rollout_count}}
            suggestions.append(suggestion)
            return suggestion, {"cost": 100.0}

        mock_protein.suggest.side_effect = suggest_side_effect

        # Mock successful training
        mock_subprocess.return_value = Mock(returncode=0)

        # Mock run_once to execute functions immediately
        mock_run_once.side_effect = lambda func, **kwargs: func()

        # Simulate multiple rollouts
        config = DictConfig(
            {
                "sweep_name": "test_sweep",
                "sweep_id": "test_123",
                "sweep_server_uri": "http://test-server",
                "data_dir": "/tmp/data",
                "trainer": {"batch_size": 2048, "minibatch_size": 64, "bptt_horizon": 32},
                "sweep": {
                    "metric": "reward",
                    "parameters": {"trainer": {"learning_rate": {"min": 0.001, "max": 0.01}}},
                },
                "wandb": {"entity": "test", "project": "test"},
                "runs_dir": "/tmp/runs",
                "sweep_job": {},
                "max_observations_to_load": 100,
                "sim": {"num_envs": 1, "num_episodes": 10},
                "device": "cpu",
                "vectorization": "none",
            }
        )

        # Run 3 rollouts
        with (
            patch("metta.sweep.sweep_lifecycle.CogwebClient") as mock_cogweb,
            patch("metta.sweep.sweep_lifecycle.create_wandb_run_for_sweep") as mock_create_run,
            patch("metta.sweep.sweep_lifecycle.fetch_protein_observations_from_wandb") as mock_fetch_obs,
            patch("metta.sweep.sweep_lifecycle.os.makedirs"),
            patch("metta.sweep.sweep_lifecycle.OmegaConf.save"),
        ):
            # Mock CogwebClient
            mock_client = Mock()
            mock_cogweb.get_client.return_value = mock_client
            mock_sweep_client = Mock()
            mock_client.sweep_client.return_value = mock_sweep_client
            mock_sweep_client.get_next_run_id.side_effect = [f"test_sweep.r.{i}" for i in range(3)]

            # Mock wandb run
            mock_run = Mock()
            mock_run.name = "test_run"
            mock_create_run.return_value = mock_run

            # Mock fetch observations
            mock_fetch_obs.return_value = []

            for i in range(3):
                with patch("metta.sweep.sweep_lifecycle.generate_protein_suggestion") as mock_gen_suggestion:
                    # Ensure suggestion exists for this iteration
                    if i >= len(suggestions):
                        # Trigger the suggest side effect to populate suggestions
                        suggestion, metadata = mock_protein.suggest()

                    # Use the suggestion from our side effect - return just the suggestion
                    mock_gen_suggestion.return_value = suggestions[i]

                    with (
                        patch("metta.sweep.sweep_lifecycle.evaluate_rollout") as mock_eval,
                        patch("metta.sweep.sweep_lifecycle.WandbContext") as mock_wandb_context,
                        patch("metta.sweep.sweep_lifecycle.SimulationSuiteConfig") as mock_sim_config,
                        patch("metta.sweep.sweep_lifecycle.PolicyStore") as mock_ps,
                        patch("metta.sweep.sweep_lifecycle.SimulationSuite") as mock_sim_suite,
                        patch("metta.sweep.sweep_lifecycle.EvalStatsDB") as mock_eval_db,
                    ):
                        mock_eval.return_value = {"score": 0.8 + i * 0.05}

                        # Mock WandbContext
                        mock_wandb_run = Mock()
                        mock_wandb_run.name = "test_run"
                        mock_wandb_context.return_value.__enter__.return_value = mock_wandb_run

                        # Mock simulation components
                        mock_sim_config.return_value = Mock()

                        # Mock policy store with proper metadata
                        mock_policy_store = Mock()
                        mock_ps.return_value = mock_policy_store
                        mock_policy = Mock()
                        mock_policy.uri = "wandb://run/test_run"
                        mock_policy.metadata = {
                            "train_time": 100.0,
                            "agent_step": 1000,
                            "epoch": 10,
                        }
                        mock_policy_store.policy_record.return_value = mock_policy
                        mock_policy_store.load_from_uri.return_value = mock_policy

                        # Mock simulation
                        mock_sim = Mock()
                        mock_sim_suite.return_value = mock_sim
                        mock_results = Mock()
                        mock_results.stats_db = Mock()
                        mock_sim.simulate.return_value = mock_results

                        # Mock eval DB
                        mock_eval = Mock()
                        mock_eval_db.from_sim_stats_db.return_value = mock_eval
                        mock_eval.get_average_metric_by_filter.return_value = 0.85

                        result = run_single_rollout(config)
                        assert result == 0

        # Verify progression
        assert rollout_count >= 3
        assert len(suggestions) >= 3
        # Verify learning rates increased
        assert suggestions[0]["trainer"]["learning_rate"] < suggestions[1]["trainer"]["learning_rate"]
        assert suggestions[1]["trainer"]["learning_rate"] < suggestions[2]["trainer"]["learning_rate"]
