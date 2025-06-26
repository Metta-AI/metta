from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
from pydantic import ValidationError

from metta.rl.trainer_config import OptimizerConfig, parse_trainer_config

valid_optimizer_config = {
    "type": "adam",
    "learning_rate": 0.001,
    "beta1": 0.9,
    "beta2": 0.999,
    "eps": 1e-8,
    "weight_decay": 0.0,
}

# Complete valid trainer config with all required fields
valid_trainer_config = {
    "_target_": "metta.rl.trainer.MettaTrainer",
    "batch_size": 1024,
    "minibatch_size": 256,
    "total_timesteps": 1000000,
    "optimizer": valid_optimizer_config,
    "clip_coef": 0.1,
    "grad_mean_variance_interval": 0,
    "ent_coef": 0.01,
    "gae_lambda": 0.95,
    "gamma": 0.99,
    "max_grad_norm": 0.5,
    "vf_coef": 0.5,
    "vf_clip_coef": 0.1,
    "l2_reg_loss_coef": 0,
    "l2_init_loss_coef": 0,
    "bptt_horizon": 32,
    "update_epochs": 1,
    "forward_pass_minibatch_target_size": 512,
    "env": "/env/mettagrid/simple",
    "norm_adv": True,
    "clip_vloss": True,
    "target_kl": None,
    "zero_copy": True,
    "require_contiguous_env_ids": False,
    "verbose": True,
    "cpu_offload": False,
    "compile": False,
    "compile_mode": "reduce-overhead",
    "profiler_interval_epochs": 10000,
    "async_factor": 2,
    "evaluate_interval": 300,
    "checkpoint_interval": 60,
    "wandb_checkpoint_interval": 300,
    "replay_interval": 300,
    "num_workers": 1,
    # Nested configs that were removed from defaults
    "lr_scheduler": {
        "enabled": False,
        "anneal_lr": False,
        "warmup_steps": None,
        "schedule_type": None,
    },
    "prioritized_experience_replay": {
        "prio_alpha": 0.0,
        "prio_beta0": 0.6,
    },
    "vtrace": {
        "vtrace_rho_clip": 1.0,
        "vtrace_c_clip": 1.0,
    },
    "kickstart": {
        "teacher_uri": None,
        "action_loss_coef": 1.0,
        "value_loss_coef": 1.0,
        "anneal_ratio": 0.65,
        "kickstart_steps": 1_000_000_000,
        "additional_teachers": None,
    },
    "env_overrides": {},
    "initial_policy": {
        "uri": None,
        "type": "top",
        "range": 1,
        "metric": "epoch",
        "filters": {},
    },
}


class TestTypedConfigs:
    def test_basic_typed_config_parsing(self):
        trainer_config = parse_trainer_config(DictConfig(valid_trainer_config))
        assert trainer_config.optimizer.type == "adam"
        assert trainer_config.optimizer.learning_rate == 0.001
        assert trainer_config.bptt_horizon == 32

    def test_config_field_validation(self):
        # invalid field
        with pytest.raises(ValidationError) as err:
            _ = OptimizerConfig.model_validate({**valid_optimizer_config, "learning_rate": -1.0})
        assert "learning_rate" in str(err)

        # invalid field
        with pytest.raises(ValidationError) as err:
            _ = OptimizerConfig.model_validate({**valid_optimizer_config, "beta1": 1.5})
        assert "beta1" in str(err)

    def test_config_missing_or_extra_field(self):
        # missing field
        missing_field_config = valid_optimizer_config.copy()
        del missing_field_config["learning_rate"]
        with pytest.raises(ValidationError) as err:
            _ = OptimizerConfig.model_validate(missing_field_config)
        assert "learning_rate" in str(err)

        # extra field
        with pytest.raises(ValidationError) as err:
            _ = OptimizerConfig.model_validate({**valid_optimizer_config, "extra_field": "extra_value"})
        assert "extra_field" in str(err)

    def test_trainer_config_all_fields_required(self):
        """Test that all fields are now required (no in-code defaults)."""
        # Try to create config with missing optimizer fields
        incomplete_config = valid_trainer_config.copy()
        incomplete_config["optimizer"] = {
            "type": "adam",
            "learning_rate": 0.001,
            # Missing beta1, beta2, eps, weight_decay
        }

        with pytest.raises(ValidationError) as err:
            parse_trainer_config(DictConfig(incomplete_config))

        # Should complain about missing optimizer fields
        assert "beta1" in str(err.value)
        assert "beta2" in str(err.value)
        assert "eps" in str(err.value)
        assert "weight_decay" in str(err.value)

    def test_trainer_config_to_dictconfig_conversion(self):
        """Test that TrainerConfig fields can be converted back to DictConfig without issues.

        This is important because in some parts of the codebase we need to create new DictConfig
        objects from subparts of a validated TrainerConfig (e.g., when passing env_overrides
        to environment constructors or when using hydra.utils.instantiate).
        """
        # Create a test config with env_overrides and kickstart
        test_config_dict = {
            **valid_trainer_config,
            "env_overrides": {
                "desync_episodes": True,
                "max_steps": 1000,
                "num_agents": 4,
            },
            "kickstart": {
                "teacher_uri": None,
                "action_loss_coef": 1.0,
                "value_loss_coef": 1.0,
                "anneal_ratio": 0.65,
                "kickstart_steps": 1_000_000_000,
                "additional_teachers": [],
            },
        }
        test_config = DictConfig(test_config_dict)

        validated_config = parse_trainer_config(test_config)

        # Test that env_overrides can be converted to DictConfig
        env_overrides_dict = DictConfig(validated_config.env_overrides)
        assert isinstance(env_overrides_dict, DictConfig)
        assert env_overrides_dict.desync_episodes is True
        assert env_overrides_dict.max_steps == 1000
        assert env_overrides_dict.num_agents == 4

        # Test that we can convert the entire config back to dict for hydra.utils.instantiate
        config_dict = validated_config.model_dump(by_alias=True)
        assert config_dict["_target_"] == "metta.rl.trainer.MettaTrainer"
        assert config_dict["batch_size"] == 1024
        assert config_dict["env_overrides"]["desync_episodes"] is True
        assert config_dict["env_overrides"]["max_steps"] == 1000

        # Test that nested configs also work
        assert isinstance(validated_config.optimizer, OptimizerConfig)
        assert validated_config.optimizer.type == "adam"
        assert validated_config.optimizer.learning_rate == 0.001

        # Test that the kickstart config works
        assert isinstance(validated_config.kickstart.additional_teachers, list)
        assert validated_config.kickstart.teacher_uri is None
        assert validated_config.kickstart.anneal_ratio == 0.65


configs_dir = str(Path(__file__).parent.parent.parent / "configs" / "trainer")


def load_config_with_hydra(trainer_name: str, overrides: list[str] | None = None) -> DictConfig:
    configs_dir = str(Path(__file__).parent.parent.parent / "configs")
    default_overrides = [
        "run=test_run",
        f"trainer={trainer_name}",
        "wandb=off",  # Disable wandb for tests
    ]

    with initialize_config_dir(config_dir=configs_dir, version_base=None):
        return compose(
            config_name="train_job",
            overrides=default_overrides + (overrides or []),
        )


class TestRealTypedConfigs:
    def test_all_trainer_configs_comprehensive(self):
        config_files = [f.stem for f in Path(configs_dir).glob("*.yaml")]

        for config_name in config_files:
            try:
                # some of the configs don't have num_workers specified because train.sh provides it
                cfg = load_config_with_hydra(config_name, overrides=["trainer.num_workers=1"])

                validated_config = parse_trainer_config(cfg.trainer)

                # Verify some basic fields and  constraints
                assert validated_config.batch_size > 0
                assert validated_config.batch_size >= validated_config.minibatch_size
                assert validated_config.batch_size % validated_config.minibatch_size == 0
                assert 0 < validated_config.gamma <= 1
                assert 0 <= validated_config.gae_lambda <= 1
                assert 0 < validated_config.optimizer.learning_rate <= 1
            except Exception as e:
                print(f"Error loading config {config_name}: {e}")
                raise e
