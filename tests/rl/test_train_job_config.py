from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from metta.common.wandb.wandb_context import WandbConfigOff, WandbConfigOn
from metta.rl.train_job_config import parse_train_job_config
from metta.rl.trainer_config import TrainerConfig
from metta.sim.simulation_config import SimulationSuiteConfig

configs_dir = str(Path(__file__).parent.parent.parent / "configs")


def test_train_job_config_parsing():
    """Test that train job configs can be parsed."""
    with initialize_config_dir(config_dir=configs_dir, version_base=None):
        cfg = compose(
            config_name="train_job",
            overrides=["run=test_run", "wandb=off", "trainer.num_workers=2"],
        )

        config = parse_train_job_config(cfg)

        assert config.run == "test_run"
        assert config.device == "cuda"
        assert config.run_dir == "./train_dir/test_run"
        assert config.policy_uri == "file://./train_dir/test_run/checkpoints"
        assert isinstance(config.trainer, TrainerConfig)
        assert config.trainer.batch_size > 0
        assert isinstance(config.wandb, (WandbConfigOff, WandbConfigOn))
        assert not config.wandb.enabled
        assert isinstance(config.train_job.evals, SimulationSuiteConfig)


def test_all_config_combinations():
    """Test various config file combinations."""
    # Test with different agent configs
    agent_configs = ["fast", "reference_design", "latent_attn_small"]

    for agent_config in agent_configs:
        with initialize_config_dir(config_dir=configs_dir, version_base=None):
            cfg = compose(
                config_name="train_job",
                overrides=[
                    "run=test_run",
                    f"agent={agent_config}",
                    "wandb=off",
                    "trainer.num_workers=1",
                ],
            )

            config = parse_train_job_config(cfg)
            assert config.agent is not None
            assert isinstance(config.agent, DictConfig)


def test_wandb_enabled_config():
    """Test parsing when wandb is enabled."""
    with initialize_config_dir(config_dir=configs_dir, version_base=None):
        cfg = compose(
            config_name="train_job",
            overrides=[
                "run=test_run",
                "trainer.num_workers=2",
                "wandb.entity=test_entity",
                "wandb.project=test_project",
                "wandb.group=test_group",
            ],
        )

        config = parse_train_job_config(cfg)
        assert config.wandb.enabled is True
        assert config.wandb.entity == "test_entity"
        assert config.wandb.project == "test_project"
        assert config.wandb.group == "test_group"


def test_config_field_types():
    """Test that all fields have the correct types."""
    with initialize_config_dir(config_dir=configs_dir, version_base=None):
        cfg = compose(
            config_name="train_job",
            overrides=["run=test_run", "wandb=off", "trainer.num_workers=2"],
        )

        config = parse_train_job_config(cfg)

        # Check basic field types
        assert isinstance(config.run, str)
        assert isinstance(config.run_dir, str)
        assert isinstance(config.data_dir, str)
        assert isinstance(config.policy_uri, str)
        assert isinstance(config.device, str)
        assert isinstance(config.seed, int)
        assert isinstance(config.policy_cache_size, int)
        assert isinstance(config.stats_server_uri, str)
        assert isinstance(config.torch_deterministic, bool)
        assert isinstance(config.vectorization, str)
        assert isinstance(config.cmd, str)

        # Check nested config types
        assert isinstance(config.trainer, TrainerConfig)
        assert isinstance(config.wandb, (WandbConfigOff, WandbConfigOn))
        assert isinstance(config.train_job.evals, SimulationSuiteConfig)

        # Check optional fields
        assert config.dist_cfg_path is None or isinstance(config.dist_cfg_path, str)
        assert config.agent is None or isinstance(config.agent, DictConfig)
        assert config.pytorch is None or isinstance(config.pytorch, DictConfig)


def test_trainer_config_runtime_paths():
    """Test that trainer config runtime paths are set correctly."""
    with initialize_config_dir(config_dir=configs_dir, version_base=None):
        cfg = compose(
            config_name="train_job",
            overrides=["run=test_run", "wandb=off", "trainer.num_workers=2"],
        )

        config = parse_train_job_config(cfg)

        # Check that runtime paths were set
        assert config.trainer.checkpoint.checkpoint_dir == "./train_dir/test_run/checkpoints"
        assert config.trainer.simulation.replay_dir == "s3://softmax-public/replays/test_run"
        # profile_dir is set in the config file directly
        assert config.trainer.profiler.profile_dir == "s3://softmax-public/torch_traces/test_run"


def test_config_validation():
    """Test that config validation works correctly."""
    with initialize_config_dir(config_dir=configs_dir, version_base=None):
        cfg = compose(
            config_name="train_job",
            overrides=[
                "run=test_run",
                "wandb=off",
                "trainer.num_workers=2",
                "trainer.minibatch_size=1000",
                "trainer.batch_size=500",  # Invalid: batch_size < minibatch_size
            ],
        )

        # This should raise a validation error
        with pytest.raises(ValueError, match="minibatch_size must be <= batch_size"):
            parse_train_job_config(cfg)


def test_backward_compatibility():
    """Test that old config format still works."""
    # Create a minimal config dict
    cfg_dict = {
        "run": "test_run",
        "run_dir": "./train_dir/test_run",
        "data_dir": "./train_dir",
        "policy_uri": "file://./train_dir/test_run/checkpoints",
        "device": "cuda",
        "seed": 0,
        "policy_cache_size": 10,
        "stats_server_uri": "https://api.observatory.softmax-research.net",
        "torch_deterministic": True,
        "vectorization": "multiprocessing",
        "cmd": "train",
        "trainer": {
            "_target_": "metta.rl.trainer.MettaTrainer",
            "num_workers": 2,
            "batch_size": 1024,
            "minibatch_size": 256,
        },
        "wandb": {"enabled": False},
        "train_job": {
            "evals": {
                "name": "test_suite",
                "num_episodes": 10,
                "simulations": {},
            }
        },
    }

    # Convert to DictConfig
    cfg = DictConfig(cfg_dict)

    # Should parse without errors
    config = parse_train_job_config(cfg)
    assert config.run == "test_run"
    assert config.trainer.num_workers == 2
