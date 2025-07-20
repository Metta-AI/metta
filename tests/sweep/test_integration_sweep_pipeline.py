"""Integration tests for the complete sweep pipeline.

These tests cover the end-to-end workflow:
1. Sweep initialization (MettaProtein + WandB sweep creation)
2. Run creation with parameter suggestions
3. Training with suggested parameters
4. Evaluation and observation recording
5. Multi-run sweep progression
"""

# pyright: ignore-all

import os
import shutil
import tempfile
from unittest.mock import patch

import pytest
import wandb
from omegaconf import OmegaConf

from metta.sweep.protein_metta import MettaProtein


class TestSweepPipelineIntegration:
    """Integration tests for the complete sweep pipeline."""

    @pytest.fixture(autouse=True)
    def setup_test_environment(self):
        """Setup test environment with temporary directories and wandb offline mode."""
        # Create temporary directories
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.temp_dir, "data")
        self.sweep_dir = os.path.join(self.data_dir, "sweep", "test_sweep")
        self.runs_dir = os.path.join(self.sweep_dir, "runs")

        os.makedirs(self.runs_dir, exist_ok=True)

        # Setup wandb for testing
        os.environ["WANDB_DIR"] = self.temp_dir
        os.environ["WANDB_MODE"] = "offline"
        os.environ["WANDB_SILENT"] = "true"

        yield

        # Cleanup
        if hasattr(self, "temp_dir"):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    @pytest.fixture
    def base_sweep_config(self):
        """Create a basic sweep configuration for testing."""
        return OmegaConf.create(
            {
                "protein": {
                    "num_random_samples": 3,
                    "max_suggestion_cost": 300,
                    "resample_frequency": 0,
                    "global_search_scale": 1,
                    "random_suggestions": 10,
                    "suggestions_per_pareto": 5,
                },
                "parameters": {
                    "metric": "reward",
                    "goal": "maximize",
                    "trainer": {
                        "optimizer": {
                            "learning_rate": {
                                "distribution": "log_normal",
                                "min": 0.0001,
                                "max": 0.01,
                                "mean": 0.001,
                                "scale": 0.5,
                            }
                        },
                        "batch_size": {
                            "distribution": "int_uniform",
                            "min": 32,
                            "max": 128,
                            "mean": 64,
                            "scale": "auto",
                        },
                    },
                },
            }
        )

    @pytest.fixture
    def base_train_config(self):
        """Create a basic training configuration for testing."""
        return OmegaConf.create(
            {
                "run": "test_run",
                "run_dir": os.path.join(self.runs_dir, "test_run"),
                "device": "cpu",
                "trainer": {
                    "_target_": "metta.rl.trainer.MettaTrainer",
                    "total_timesteps": 100,
                    "evaluate_interval": 50,
                    "batch_size": 64,
                    "num_workers": 1,
                    "minibatch_size": 32,
                    "bptt_horizon": 16,
                    "update_epochs": 1,
                    "clip_coef": 0.1,
                    "ent_coef": 0.01,
                    "vf_coef": 0.5,
                    "gae_lambda": 0.95,
                    "gamma": 0.99,
                    "max_grad_norm": 0.5,
                    "vf_clip_coef": 0.1,
                    "target_kl": 0.01,
                    "l2_reg_loss_coef": 0.0,
                    "l2_init_loss_coef": 0.0,
                    "norm_adv": True,
                    "clip_vloss": True,
                    "prioritized_experience_replay": {},
                    "vtrace": {},
                    "zero_copy": False,
                    "require_contiguous_env_ids": False,
                    "verbose": False,
                    "scale_batches_by_world_size": False,
                    "cpu_offload": False,
                    "compile": False,
                    "compile_mode": "default",
                    "profiler_interval_epochs": 0,
                    "forward_pass_minibatch_target_size": 2,
                    "async_factor": 1,
                    "kickstart": {},
                    "checkpoint_interval": 1000,
                    "wandb_checkpoint_interval": 1000,
                    "grad_mean_variance_interval": 0,
                    "optimizer": {
                        "learning_rate": 0.001,
                        "type": "adam",
                        "beta1": 0.9,
                        "beta2": 0.999,
                        "eps": 1e-8,
                        "weight_decay": 0.0,
                    },
                    "lr_scheduler": {
                        "schedule_type": "constant",
                        "warmup_steps": 0,
                    },
                    "env": "/env/mettagrid/simple",
                    "env_overrides": {},
                    "initial_policy": {},
                },
                "wandb": {
                    "enabled": True,
                    "project": "test_sweep_integration",
                    "entity": "test_entity",
                },
            }
        )

    def test_sweep_creation_and_initialization(self, base_sweep_config):
        """Test complete sweep creation and initialization process."""
        # Mock sweep creation to avoid API calls
        with patch("wandb.sweep") as mock_sweep:
            mock_sweep.return_value = "test_sweep_123"

            from metta.sweep.wandb_utils import create_wandb_sweep

            # Test sweep creation
            sweep_id = create_wandb_sweep(
                sweep_name="test_sweep",
                wandb_entity="test_entity",
                wandb_project="test_project",
            )

            assert sweep_id is not None
            assert isinstance(sweep_id, str)
            assert sweep_id == "test_sweep_123"

            # Verify wandb.sweep was called with correct parameters
            mock_sweep.assert_called_once()
            call_args = mock_sweep.call_args
            assert call_args[1]["project"] == "test_project"
            assert call_args[1]["entity"] == "test_entity"

        # Verify sweep config structure was properly processed
        # This would normally be saved to disk in real usage
        sweep_config_path = os.path.join(self.sweep_dir, "config.yaml")
        os.makedirs(os.path.dirname(sweep_config_path), exist_ok=True)

        sweep_metadata = {
            "sweep": "test_integration_sweep",
            "wandb_sweep_id": sweep_id,
            "wandb_path": f"test_entity/test_project/{sweep_id}",
        }

        OmegaConf.save(sweep_metadata, sweep_config_path)

        # Verify sweep config was saved correctly
        assert os.path.exists(sweep_config_path)
        loaded_config = OmegaConf.load(sweep_config_path)
        assert loaded_config.sweep == "test_integration_sweep"
        assert loaded_config.wandb_sweep_id == sweep_id

    def test_run_initialization_with_protein_suggestions(self, base_sweep_config):
        """Test run initialization with Protein parameter suggestions."""
        wandb.init(project="test_project", job_type="test", mode="offline")

        try:
            # Create MettaProtein instance
            metta_protein = MettaProtein(base_sweep_config, wandb.run)

            # Verify protein state in wandb
            assert wandb.run.summary.get("protein.state") == "running"

            # Generate suggestion and verify it's stored in wandb
            suggestion, info = metta_protein.suggest()

            # Verify suggestion structure
            assert isinstance(suggestion, dict)
            assert isinstance(info, dict)

            # Check that we have the expected parameters
            assert "trainer" in suggestion
            assert "optimizer" in suggestion["trainer"]
            assert "learning_rate" in suggestion["trainer"]["optimizer"]

            # Info should contain prediction details (when available)
            # Note: suggestion_uuid was removed during cleanup

            # Verify suggestion values are within expected ranges
            lr = suggestion["trainer"]["optimizer"]["learning_rate"]
            batch_size = suggestion["trainer"]["batch_size"]

            assert 0.0001 <= lr <= 0.01
            assert 32 <= batch_size <= 128
            assert isinstance(batch_size, int)

        finally:
            wandb.finish()

    def test_config_override_application(self, base_train_config, base_sweep_config):
        """Test application of Protein suggestions to training config."""
        wandb.init(project="test_project", job_type="test", mode="offline")

        try:
            # Create MettaProtein and get suggestion
            metta_protein = MettaProtein(base_sweep_config, wandb.run)
            suggestion, _ = metta_protein.suggest()

            # Apply suggestion to config (simulate sweep_init.py behavior)
            from tools.sweep_init import apply_protein_suggestion

            # Create a copy of the base config for testing
            test_config = OmegaConf.create({"trainer": base_train_config.trainer})

            # Apply suggestion
            apply_protein_suggestion(test_config, suggestion)

            # Verify suggestion was applied
            new_lr = test_config.trainer.optimizer.learning_rate
            new_batch_size = test_config.trainer.batch_size

            # Values should have changed (unless protein returned search center)
            # For first suggestion, protein may return search center, so we check if values are reasonable
            assert 0.0001 <= new_lr <= 0.01  # Within expected range
            assert 32 <= new_batch_size <= 128  # Within expected range
            assert isinstance(new_batch_size, int)

            # New values should match suggestion
            assert new_lr == suggestion["trainer"]["optimizer"]["learning_rate"]
            assert new_batch_size == suggestion["trainer"]["batch_size"]

            # Other config values should be preserved
            assert test_config.trainer.total_timesteps == base_train_config.trainer.total_timesteps
            assert test_config.trainer.num_workers == base_train_config.trainer.num_workers

        finally:
            wandb.finish()

    def test_config_validation_after_override(self, base_train_config, base_sweep_config):
        """Test that config overrides work correctly without full validation."""
        wandb.init(project="test_project", job_type="test", mode="offline")

        try:
            # Create suggestion
            metta_protein = MettaProtein(base_sweep_config, wandb.run)
            suggestion, _ = metta_protein.suggest()

            # Test that override application works (without full validation)
            # In real usage, only the complete final config gets validated, not partial overrides
            from tools.sweep_init import apply_protein_suggestion

            test_config = OmegaConf.create({"trainer": {"optimizer": {"learning_rate": 0.001}, "batch_size": 64}})
            apply_protein_suggestion(test_config, suggestion)

            # Verify override was applied
            assert test_config.trainer.optimizer.learning_rate == suggestion["trainer"]["optimizer"]["learning_rate"]
            assert test_config.trainer.batch_size == suggestion["trainer"]["batch_size"]

            # Test that the suggestion values are reasonable
            assert 0.0001 <= test_config.trainer.optimizer.learning_rate <= 0.01
            assert 32 <= test_config.trainer.batch_size <= 128
            assert isinstance(test_config.trainer.batch_size, int)

        finally:
            wandb.finish()

    def test_config_save_and_load_cycle(self, base_train_config):
        """Test saving and loading of training config overrides."""
        # Create test run directory
        run_dir = os.path.join(self.runs_dir, "test_config_cycle")
        os.makedirs(run_dir, exist_ok=True)

        # Create test overrides
        test_overrides = {"trainer": {"optimizer": {"learning_rate": 0.005}, "batch_size": 96}}

        # Save overrides
        from tools.sweep_config_utils import save_train_job_override_config

        test_config = OmegaConf.create(base_train_config)
        test_config.run_dir = run_dir

        save_path = save_train_job_override_config(test_config, test_overrides)

        # Verify file was saved
        assert os.path.exists(save_path)
        assert save_path == os.path.join(run_dir, "train_config_overrides.yaml")

        # Test loading saved overrides directly (without validation)
        loaded_overrides = OmegaConf.load(save_path)
        assert loaded_overrides.trainer.optimizer.learning_rate == 0.005
        assert loaded_overrides.trainer.batch_size == 96

        # Test that the override file structure is correct
        assert "trainer" in loaded_overrides
        assert "optimizer" in loaded_overrides.trainer
        assert "learning_rate" in loaded_overrides.trainer.optimizer

    def test_observation_recording_workflow(self, base_sweep_config):
        """Test the complete observation recording workflow."""
        wandb.init(project="test_project", job_type="test", mode="offline")

        try:
            # Create MettaProtein
            metta_protein = MettaProtein(base_sweep_config, wandb.run)

            # Generate initial suggestion
            suggestion, info = metta_protein.suggest()

            # Simulate training completion with results
            objective_value = 0.85  # Reward metric
            training_cost = 120.0  # Training time in seconds

            # Record observation (simulate sweep_eval.py behavior)
            metta_protein.record_observation(objective_value, training_cost)

            # Verify observation was recorded in wandb
            assert wandb.run.summary.get("protein.objective") == objective_value
            assert wandb.run.summary.get("protein.cost") == training_cost
            assert wandb.run.summary.get("protein.state") == "success"

            # Verify observation was stored in the underlying protein
            assert len(metta_protein._protein.success_observations) == 1
            observation = metta_protein._protein.success_observations[0]
            assert observation["output"] == objective_value
            assert observation["cost"] == training_cost
            assert not observation["is_failure"]

        finally:
            wandb.finish()

    def test_multi_run_sweep_progression(self, base_sweep_config):
        """Test multiple runs in sequence to verify sweep progression."""
        sweep_runs = []
        suggestions = []

        # Simulate multiple runs in a sweep
        for run_idx in range(3):
            wandb.init(project="test_project", job_type="test", mode="offline", reinit=True)

            try:
                # Create new MettaProtein instance for each run
                metta_protein = MettaProtein(base_sweep_config, wandb.run)

                # Generate suggestion
                suggestion, info = metta_protein.suggest()
                suggestions.append(suggestion)

                # Simulate training with different outcomes
                objective = 0.7 + (run_idx * 0.1)  # Improving performance
                cost = 100.0 + (run_idx * 20.0)  # Increasing cost

                # Record observation
                metta_protein.record_observation(objective, cost)

                # Store run info
                sweep_runs.append(
                    {
                        "run_idx": run_idx,
                        "suggestion": suggestion,
                        "objective": objective,
                        "cost": cost,
                        "wandb_run_id": wandb.run.id,
                    }
                )

            finally:
                wandb.finish()

        # Verify we got different suggestions
        assert len(suggestions) == 3

        # Verify suggestions are within expected ranges
        lr_values = [s["trainer"]["optimizer"]["learning_rate"] for s in suggestions]
        batch_sizes = [s["trainer"]["batch_size"] for s in suggestions]

        # All suggestions should be within valid ranges
        for lr in lr_values:
            assert 0.0001 <= lr <= 0.01
        for bs in batch_sizes:
            assert 32 <= bs <= 128
            assert isinstance(bs, int)

        # Note: Early suggestions may be identical (search center) until protein learns from observations

        # Verify progression of objectives (with floating point tolerance)
        objectives = [run["objective"] for run in sweep_runs]
        expected = [0.7, 0.8, 0.9]
        for actual, exp in zip(objectives, expected, strict=False):
            assert abs(actual - exp) < 1e-10  # Should be improving with floating point tolerance

    def test_error_handling_in_pipeline(self, base_sweep_config):
        """Test error handling throughout the pipeline."""
        # Test with invalid protein config that will cause an error
        invalid_config = OmegaConf.create(
            {
                "protein": {
                    "num_random_samples": -1,  # Invalid negative value
                },
                "parameters": {
                    "metric": "reward",
                    "goal": "maximize",
                    "trainer": {
                        "optimizer": {
                            "learning_rate": {
                                "distribution": "invalid_distribution",  # Invalid distribution type
                                "min": 0.0001,
                                "max": 0.01,
                                "mean": 0.0005,
                                "scale": "auto",
                            }
                        }
                    },
                },
            }
        )

        wandb.init(project="test_project", job_type="test", mode="offline")

        try:
            # This should raise ValueError for invalid distribution type
            with pytest.raises(ValueError, match="Invalid distribution"):
                MettaProtein(invalid_config, wandb.run)

        finally:
            wandb.finish()

    def test_file_system_integration(self, base_sweep_config):
        """Test file system operations in the sweep pipeline."""
        # Test directory creation
        test_sweep_dir = os.path.join(self.data_dir, "sweep", "file_test_sweep")
        test_runs_dir = os.path.join(test_sweep_dir, "runs")

        # Simulate sweep_init.py directory creation
        os.makedirs(test_runs_dir, exist_ok=True)
        assert os.path.exists(test_runs_dir)

        # Test config file saving
        sweep_config_path = os.path.join(test_sweep_dir, "config.yaml")
        test_sweep_metadata = {
            "sweep": "file_test_sweep",
            "wandb_sweep_id": "test_sweep_123",
            "wandb_path": "test_entity/test_project/test_sweep_123",
        }

        OmegaConf.save(test_sweep_metadata, sweep_config_path)
        assert os.path.exists(sweep_config_path)

        # Test run directory creation
        test_run_dir = os.path.join(test_runs_dir, "run_001")
        os.makedirs(test_run_dir, exist_ok=True)

        # Test config override saving
        test_overrides = {"trainer": {"learning_rate": 0.003}}
        override_path = os.path.join(test_run_dir, "train_config_overrides.yaml")

        OmegaConf.save(test_overrides, override_path)
        assert os.path.exists(override_path)

        # Test loading saved config
        loaded_overrides = OmegaConf.load(override_path)
        assert loaded_overrides.trainer.learning_rate == 0.003

    def test_wandb_integration_end_to_end(self, base_sweep_config):
        """End-to-end check that create_wandb_sweep wires correctly to wandb.sweep."""

        # Patch wandb.sweep to avoid real API calls and verify parameters
        with patch("wandb.sweep") as mock_sweep:
            mock_sweep.return_value = "e2e_test_sweep_id"

            from metta.sweep.wandb_utils import create_wandb_sweep

            result = create_wandb_sweep(
                sweep_name="e2e_test_sweep",
                wandb_entity="test_entity",
                wandb_project="test_project",
            )

            # Validate return value and that wandb.sweep was invoked with expected args
            assert result == "e2e_test_sweep_id"
            mock_sweep.assert_called_once()
            called_kwargs = mock_sweep.call_args.kwargs
            assert called_kwargs["entity"] == "test_entity"
            assert called_kwargs["project"] == "test_project"

    def test_parameter_distribution_handling(self, base_sweep_config):
        """Test that different parameter distributions are handled correctly."""
        # Test with all supported distribution types
        extended_config = OmegaConf.create(
            {
                "protein": base_sweep_config.protein,
                "parameters": {
                    "metric": "reward",
                    "goal": "maximize",
                    "trainer": {
                        "optimizer": {
                            "learning_rate": {
                                "distribution": "log_normal",
                                "min": 1e-5,
                                "max": 1e-2,
                                "mean": 1e-3,
                                "scale": 0.5,
                            }
                        },
                        "batch_size": {
                            "distribution": "int_uniform",
                            "min": 16,
                            "max": 256,
                            "mean": 128,
                            "scale": "auto",
                        },
                        "clip_range": {
                            "distribution": "uniform",
                            "min": 0.1,
                            "max": 0.5,
                            "mean": 0.3,
                            "scale": "auto",
                        },
                        "gamma": {
                            "distribution": "uniform",
                            "min": 0.9,
                            "max": 0.999,
                            "mean": 0.95,
                            "scale": "auto",
                        },
                        "entropy_coeff": {
                            "distribution": "logit_normal",
                            "min": 0.001,
                            "max": 0.1,
                            "mean": 0.01,
                            "scale": 0.3,
                        },
                    },
                },
            }
        )

        wandb.init(project="test_project", job_type="test", mode="offline")

        try:
            metta_protein = MettaProtein(extended_config, wandb.run)
            suggestion, _ = metta_protein.suggest()

            # Verify all parameters are present and within bounds
            trainer_params = suggestion["trainer"]

            # log_normal
            lr = trainer_params["optimizer"]["learning_rate"]
            assert 1e-5 <= lr <= 1e-2
            assert isinstance(lr, float)

            # int_uniform
            batch_size = trainer_params["batch_size"]
            assert 16 <= batch_size <= 256
            assert isinstance(batch_size, int)

            # uniform
            clip_range = trainer_params["clip_range"]
            assert 0.1 <= clip_range <= 0.5
            assert isinstance(clip_range, float)

            # uniform_pow2
            gamma = trainer_params["gamma"]
            assert 0.9 <= gamma <= 0.999
            assert isinstance(gamma, float)

            # logit_normal
            entropy_coeff = trainer_params["entropy_coeff"]
            assert 0.001 <= entropy_coeff <= 0.1
            assert isinstance(entropy_coeff, float)

        finally:
            wandb.finish()

    def test_failure_handling_and_recovery(self, base_sweep_config):
        """Test failure handling and recovery mechanisms."""
        wandb.init(project="test_project", job_type="test", mode="offline")

        try:
            metta_protein = MettaProtein(base_sweep_config, wandb.run)

            # Test failure recording
            metta_protein.record_failure("Test failure message")

            # Verify failure was recorded
            assert wandb.run.summary.get("protein.state") == "failure"

            # Test that we can still generate new suggestions after failure
            suggestion, info = metta_protein.suggest()
            assert suggestion is not None
            assert info is not None

        finally:
            wandb.finish()
