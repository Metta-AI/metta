"""Test that sweep_job configuration can be loaded and used for training."""

import os
import tempfile
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from metta.rl.trainer_config import parse_trainer_config
from tools.sweep_config_utils import load_train_job_config_with_overrides


class TestSweepConfigLoading:
    """Test sweep configuration loading and validation."""

    def test_sweep_job_as_training_override_simple(self):
        """Test that a simplified sweep config can be used as training overrides."""
        # Create a simple sweep-like config without complex interpolations
        sweep_overrides = {
            "trainer": {
                "_target_": "metta.rl.trainer.MettaTrainer",
                "total_timesteps": 6400,
                "batch_size": 3200,
                "minibatch_size": 1600,
                "num_workers": 1,
                "curriculum": "/env/mettagrid/curriculum/simple",
                "checkpoint": {
                    "checkpoint_interval": 1000,
                    "wandb_checkpoint_interval": 1000,
                },
                "simulation": {
                    "evaluate_interval": 1000,
                },
                "env_overrides": {
                    "game": {
                        "max_steps": 50,
                        "num_agents": 6,
                    }
                },
                "optimizer": {
                    "learning_rate": 0.0005,
                },
                "ppo": {
                    "ent_coef": 0.01,
                },
            },
            "agent": {
                "_target_": "metta.agent.metta_agent.MettaAgent",
                "num_layers": 2,
                "num_heads": 4,
                "latent_dim": 128,
            },
            "train_job": {
                "evals": {
                    "name": "sweep_eval",
                    "num_episodes": 5,
                    "max_time_s": 300,
                    "env_overrides": {},
                    "simulations": {"simple": {"env": "env/mettagrid/simple"}},
                }
            },
            "wandb": {
                "enabled": True,
                "project": "metta-research",
                "entity": "softmax",
            },
            "run": "test_sweep.r.0",
            "seed": 12345,
            "sweep_run": "test_sweep",
            "device": "cpu",
        }

        # Get the configs directory
        config_dir = Path(__file__).parent.parent.parent / "configs"
        config_dir = config_dir.resolve()

        with initialize_config_dir(config_dir=str(config_dir), version_base=None):
            # Load train_job config
            train_cfg = compose(config_name="train_job")

            with tempfile.TemporaryDirectory() as temp_dir:
                # Set required fields
                train_cfg.run = "test_sweep.r.0"
                train_cfg.run_dir = temp_dir
                sweep_overrides["run_dir"] = temp_dir

                # Save overrides
                override_path = os.path.join(temp_dir, "train_config_overrides.yaml")
                OmegaConf.save(sweep_overrides, override_path)

                # Load config with overrides
                merged_cfg = load_train_job_config_with_overrides(train_cfg)

                # Verify essential fields
                assert merged_cfg.run == "test_sweep.r.0"
                assert merged_cfg.run_dir == temp_dir
                assert merged_cfg.seed == 12345
                assert merged_cfg.sweep_run == "test_sweep"
                assert merged_cfg.device == "cpu"

                # Verify trainer config is valid
                trainer_config = parse_trainer_config(merged_cfg)

                # Check trainer fields
                assert trainer_config.total_timesteps == 6400
                assert trainer_config.batch_size == 3200
                assert trainer_config.minibatch_size == 1600
                assert trainer_config.num_workers == 1
                assert trainer_config.curriculum == "/env/mettagrid/curriculum/simple"
                assert trainer_config.checkpoint.checkpoint_interval == 1000
                assert trainer_config.checkpoint.wandb_checkpoint_interval == 1000
                assert trainer_config.simulation.evaluate_interval == 1000
                assert trainer_config.optimizer.learning_rate == 0.0005
                assert trainer_config.ppo.ent_coef == 0.01

                # Verify evals config
                assert "train_job" in merged_cfg
                assert "evals" in merged_cfg.train_job
                assert merged_cfg.train_job.evals.name == "sweep_eval"
                assert merged_cfg.train_job.evals.num_episodes == 5

    def test_sweep_job_structure_compatibility(self):
        """Test that sweep_job.yaml structure is designed to work with train.py."""
        config_dir = Path(__file__).parent.parent.parent / "configs"
        config_dir = config_dir.resolve()

        with initialize_config_dir(config_dir=str(config_dir), version_base=None):
            # Load sweep_job config
            sweep_cfg = compose(config_name="sweep_job", overrides=["sweep_run=test_sweep"])

            # Check that sweep_job contains the key sections
            assert "trainer" in sweep_cfg
            assert "agent" in sweep_cfg
            assert "sim" in sweep_cfg
            assert "sweep_job" in sweep_cfg

            # Check sweep_job section has the right structure
            assert "trainer" in sweep_cfg.sweep_job
            assert "agent" in sweep_cfg.sweep_job
            assert "train_job" in sweep_cfg.sweep_job
            assert "evals" in sweep_cfg.sweep_job.train_job

            # Verify trainer overrides are present
            assert sweep_cfg.trainer.total_timesteps == 6400
            assert sweep_cfg.trainer.batch_size == 3200

    def test_protein_suggestion_application(self):
        """Test that protein suggestions can be applied to sweep config."""
        from tools.sweep_init import apply_protein_suggestion

        # Create a base config
        base_config = OmegaConf.create(
            {
                "trainer": {
                    "optimizer": {
                        "learning_rate": 0.001,
                        "beta1": 0.9,
                    },
                    "ppo": {
                        "ent_coef": 0.01,
                        "gae_lambda": 0.95,
                    },
                    "batch_size": 1024,
                },
                "agent": {
                    "num_layers": 2,
                },
            }
        )

        # Apply protein suggestions
        suggestions = {
            "trainer": {
                "optimizer": {
                    "learning_rate": 0.0005,  # Changed
                },
                "ppo": {
                    "ent_coef": 0.02,  # Changed
                    # gae_lambda unchanged
                },
                # batch_size unchanged
            }
        }

        apply_protein_suggestion(base_config, suggestions)

        # Verify suggestions were applied
        assert base_config.trainer.optimizer.learning_rate == 0.0005
        assert base_config.trainer.optimizer.beta1 == 0.9  # Unchanged
        assert base_config.trainer.ppo.ent_coef == 0.02
        assert base_config.trainer.ppo.gae_lambda == 0.95  # Unchanged
        assert base_config.trainer.batch_size == 1024  # Unchanged
        assert base_config.agent.num_layers == 2  # Unchanged
