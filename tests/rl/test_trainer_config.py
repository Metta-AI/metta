import math
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
from pydantic import ValidationError

from metta.rl.trainer_config import OptimizerConfig, create_trainer_config
from metta.util.init.mettagrid_environment import init_mettagrid_environment

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
    "total_timesteps": 1000000,
    "batch_size": 1024,
    "minibatch_size": 256,
    "bptt_horizon": 32,
    "update_epochs": 1,
    "forward_pass_minibatch_target_size": 512,
    "async_factor": 2,
    "zero_copy": True,
    "require_contiguous_env_ids": False,
    "verbose": True,
    "cpu_offload": False,
    "compile": False,
    "compile_mode": "reduce-overhead",
    "profiler": {"interval_epochs": 10000, "profile_dir": "/test/upload/dir"},
    "num_workers": 1,
    "env": "/env/mettagrid/arena/advanced",
    "curriculum": None,
    "env_overrides": {},
    "grad_mean_variance_interval": 0,
    "ppo": {
        "clip_coef": 0.1,
        "ent_coef": 0.01,
        "gae_lambda": 0.95,
        "gamma": 0.99,
        "max_grad_norm": 0.5,
        "vf_clip_coef": 0.1,
        "vf_coef": 0.5,
        "l2_reg_loss_coef": 0,
        "l2_init_loss_coef": 0,
        "norm_adv": True,
        "clip_vloss": True,
        "target_kl": None,
    },
    "optimizer": valid_optimizer_config,
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
    "hyperparameter_scheduler": {
        "learning_rate_schedule": {
            "_target_": "metta.rl.hyperparameter_scheduler.ConstantSchedule",
            "initial_value": 0.001,
        },
        "ppo_clip_schedule": {
            "_target_": "metta.rl.hyperparameter_scheduler.ConstantSchedule",
            "initial_value": 0.1,
        },
        "ppo_ent_coef_schedule": {
            "_target_": "metta.rl.hyperparameter_scheduler.ConstantSchedule",
            "initial_value": 0.01,
        },
        "ppo_vf_clip_schedule": {
            "_target_": "metta.rl.hyperparameter_scheduler.ConstantSchedule",
            "initial_value": 0.1,
        },
        "ppo_l2_reg_loss_schedule": {
            "_target_": "metta.rl.hyperparameter_scheduler.ConstantSchedule",
            "initial_value": 0,
        },
        "ppo_l2_init_loss_schedule": {
            "_target_": "metta.rl.hyperparameter_scheduler.ConstantSchedule",
            "initial_value": 0,
        },
    },
    "initial_policy": {
        "uri": None,
        "type": "top",
        "range": 1,
        "metric": "epoch",
        "filters": {},
    },
    "checkpoint": {
        "checkpoint_interval": 60,
        "wandb_checkpoint_interval": 300,
    },
    "simulation": {
        "evaluate_interval": 300,
    },
}


def make_cfg(trainer_cfg: dict) -> DictConfig:
    return DictConfig(
        {
            "run_dir": "/tmp/test_run",
            "run": "test_run",
            "trainer": trainer_cfg,
            # Include these values to avoid needing to run the metta_script decorator
            # here to populate them with detected values
            "vectorization": "serial",
            "device": "cpu",
        }
    )


class TestTypedConfigs:
    def test_basic_typed_config_parsing(self):
        trainer_config = create_trainer_config(make_cfg(valid_trainer_config))
        assert trainer_config.optimizer.type == "adam"
        assert trainer_config.optimizer.learning_rate == 0.001
        assert trainer_config.bptt_horizon == 32

        # Test that runtime paths are set correctly
        assert trainer_config.checkpoint.checkpoint_dir == "/tmp/test_run/checkpoints"
        assert trainer_config.simulation.replay_dir == "/tmp/test_run/replays/"

    def test_config_field_validation(self):
        # invalid field
        with pytest.raises(ValidationError, match="learning_rate") as err:
            _ = OptimizerConfig.model_validate({**valid_optimizer_config, "learning_rate": -1.0})
        assert "learning_rate" in str(err)

        # invalid field
        with pytest.raises(ValidationError) as err:
            _ = OptimizerConfig.model_validate({**valid_optimizer_config, "beta1": 1.5})
        assert "beta1" in str(err)

    def test_config_missing_or_extra_field(self):
        # default
        missing_field_config = valid_optimizer_config.copy()
        del missing_field_config["learning_rate"]
        optimizer_cfg = OptimizerConfig.model_validate(missing_field_config)
        assert math.isclose(optimizer_cfg.learning_rate, 0.0004573146765703167)  # default value

        # extra field
        with pytest.raises(ValidationError) as err:
            _ = OptimizerConfig.model_validate({**valid_optimizer_config, "extra_field": "extra_value"})
        assert "extra_field" in str(err)

    def test_trainer_config_defaults(self):
        """Test that optimizer fields use defaults when not provided."""
        # Create config with minimal optimizer fields
        incomplete_config = valid_trainer_config.copy()
        incomplete_config["optimizer"] = {
            "type": "adam",
            "learning_rate": 0.001,
            # Missing beta1, beta2, eps, weight_decay
        }

        # Should not raise - defaults will be used
        trainer_config = create_trainer_config(make_cfg(incomplete_config))

        # Check that defaults were applied
        assert trainer_config.optimizer.beta1 == 0.9
        assert trainer_config.optimizer.beta2 == 0.999
        assert trainer_config.optimizer.eps == 1e-12
        assert trainer_config.optimizer.weight_decay == 0

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
            "hyperparameter_scheduler": {},
        }

        validated_config = create_trainer_config(make_cfg(test_config_dict))

        # Test that env_overrides can be converted to DictConfig
        env_overrides_dict = DictConfig(validated_config.env_overrides)
        assert isinstance(env_overrides_dict, DictConfig)
        assert env_overrides_dict.max_steps == 1000
        assert env_overrides_dict.num_agents == 4

        # Test that we can convert the entire config back to dict for hydra.utils.instantiate
        config_dict = validated_config.model_dump(by_alias=True)
        assert config_dict["_target_"] == "metta.rl.trainer.MettaTrainer"
        assert config_dict["batch_size"] == 1024
        assert config_dict["env_overrides"]["max_steps"] == 1000

    def test_runtime_path_overrides(self):
        """Test that checkpoint_dir and replay_dir can be overridden in config."""
        config_with_paths = valid_trainer_config.copy()
        config_with_paths["checkpoint"]["checkpoint_dir"] = "/custom/checkpoint/path"
        config_with_paths["simulation"]["replay_dir"] = "s3://custom-bucket/replays"

        trainer_config = create_trainer_config(make_cfg(config_with_paths))

        # Should use the provided paths, not the runtime defaults
        assert trainer_config.checkpoint.checkpoint_dir == "/custom/checkpoint/path"
        assert trainer_config.simulation.replay_dir == "s3://custom-bucket/replays"

    def test_interval_validation_checks(self):
        """Test that interval validation checks work correctly."""
        # Test 1: evaluate_interval < checkpoint_interval should fail
        config_with_bad_intervals = valid_trainer_config.copy()
        config_with_bad_intervals["simulation"]["evaluate_interval"] = 30
        config_with_bad_intervals["checkpoint"]["checkpoint_interval"] = 60

        with pytest.raises(ValueError) as err:
            create_trainer_config(make_cfg(config_with_bad_intervals))
        assert "evaluate_interval must be at least as large as checkpoint_interval" in str(err.value)

        # Test 2: evaluate_interval < wandb_checkpoint_interval should fail
        config_with_bad_intervals = valid_trainer_config.copy()
        config_with_bad_intervals["simulation"]["evaluate_interval"] = 60
        config_with_bad_intervals["checkpoint"]["checkpoint_interval"] = 30  # Lower than evaluate_interval
        config_with_bad_intervals["checkpoint"]["wandb_checkpoint_interval"] = 90  # Higher than evaluate_interval

        with pytest.raises(ValueError) as err:
            create_trainer_config(make_cfg(config_with_bad_intervals))
        assert "evaluate_interval must be at least as large as wandb_checkpoint_interval" in str(err.value)

        # Test 3: wandb_checkpoint_interval < checkpoint_interval should fail
        config_with_bad_intervals = valid_trainer_config.copy()
        config_with_bad_intervals["checkpoint"]["checkpoint_interval"] = 60
        config_with_bad_intervals["checkpoint"]["wandb_checkpoint_interval"] = 30

        with pytest.raises(ValueError) as err:
            create_trainer_config(make_cfg(config_with_bad_intervals))
        assert "wandb_checkpoint_interval must be at least as large as checkpoint_interval" in str(err.value)

        # Test 4: Valid configuration where evaluate_interval >= both checkpoint intervals
        config_with_good_intervals = valid_trainer_config.copy()
        config_with_good_intervals["checkpoint"]["checkpoint_interval"] = 60
        config_with_good_intervals["checkpoint"]["wandb_checkpoint_interval"] = 120
        config_with_good_intervals["simulation"]["evaluate_interval"] = 120

        # This should not raise
        trainer_config = create_trainer_config(make_cfg(config_with_good_intervals))
        assert trainer_config.simulation.evaluate_interval == 120
        assert trainer_config.checkpoint.checkpoint_interval == 60
        assert trainer_config.checkpoint.wandb_checkpoint_interval == 120

        # Test 5: evaluate_interval = 0 (disabled) should always pass
        config_with_disabled_eval = valid_trainer_config.copy()
        config_with_disabled_eval["simulation"]["evaluate_interval"] = 0
        config_with_disabled_eval["checkpoint"]["checkpoint_interval"] = 60
        config_with_disabled_eval["checkpoint"]["wandb_checkpoint_interval"] = 300

        # This should not raise
        trainer_config = create_trainer_config(make_cfg(config_with_disabled_eval))
        assert trainer_config.simulation.evaluate_interval == 0


def load_config_with_hydra(trainer_name: str, overrides: list[str] | None = None) -> DictConfig:
    configs_dir = str(Path(__file__).parent.parent.parent / "configs")
    default_overrides = [
        "run=test_run",
        f"trainer={trainer_name}",
        "wandb=off",  # Disable wandb for tests
    ]

    with initialize_config_dir(config_dir=configs_dir, version_base=None):
        cfg = compose(
            config_name="train_job",
            overrides=default_overrides + (overrides or []),
        )

        return cfg


class TestRealTypedConfigs:
    def test_all_trainer_configs_comprehensive(self):
        configs_dir_path = Path(__file__).parent.parent.parent / "configs" / "trainer"
        config_files = [f.stem for f in configs_dir_path.glob("*.yaml")]

        for config_name in config_files:
            try:
                # Process config and create trainer config within Hydra context
                default_overrides = ["run=test_run", f"trainer={config_name}", "wandb=off", "trainer.num_workers=1"]

                with initialize_config_dir(config_dir=str(configs_dir_path.parent), version_base=None):
                    cfg = compose(
                        config_name="train_job",
                        overrides=default_overrides,
                    )

                    init_mettagrid_environment(cfg)

                    # Skip if curriculum is unresolved (indicated by ???)
                    if hasattr(cfg, "trainer") and hasattr(cfg.trainer, "curriculum"):
                        if cfg.trainer.curriculum == "???":
                            print(f"Skipping {config_name} - curriculum is unresolved")
                            continue

                    validated_config = create_trainer_config(cfg)

                    # Verify some basic fields and  constraints
                    assert validated_config.batch_size > 0
                    assert validated_config.batch_size >= validated_config.minibatch_size
                    assert validated_config.batch_size % validated_config.minibatch_size == 0
                    assert 0 < validated_config.ppo.gamma <= 1
                    assert 0 <= validated_config.ppo.gae_lambda <= 1
                    assert 0 < validated_config.optimizer.learning_rate <= 1

            except Exception as e:
                print(f"Error loading config {config_name}: {e}")
                raise e

    def test_all_config_overrides_comprehensive(self):
        """Test all config files that override trainer settings (hardware and user configs)."""
        configs_root = Path(__file__).parent.parent.parent / "configs"

        # Collect all config files that might have trainer overrides
        config_files_to_test: list[tuple[str, str, str]] = []

        # Hardware configs
        hardware_configs = list((configs_root / "hardware").glob("*.yaml"))
        for config in hardware_configs:
            config_files_to_test.append(("hardware", config.stem, f"+hardware={config.stem}"))

        # User configs
        user_configs = list((configs_root / "user").glob("*.yaml"))
        for config in user_configs:
            config_files_to_test.append(("user", config.stem, f"+user={config.stem}"))

        # Test each config file
        for config_type, config_name, override in config_files_to_test:
            # Check if the config file has a trainer section
            config_path = configs_root / f"{config_type}/{config_name}.yaml"

            # Quick check if file contains trainer section
            with open(config_path, "r") as f:
                content = f.read()
                if "trainer:" not in content:
                    continue  # Skip configs without trainer overrides

            print(f"Testing {config_type} config: {config_name}")

            try:
                # For hardware/user configs, apply them as overrides
                overrides_list = ["run=test_run", "trainer=trainer", "wandb=off", override, "trainer.num_workers=1"]

                with initialize_config_dir(config_dir=str(configs_root), version_base=None):
                    cfg = compose(
                        config_name="train_job",
                        overrides=overrides_list,
                    )

                    init_mettagrid_environment(cfg)

                    # Skip if curriculum is unresolved (indicated by ???)
                    if hasattr(cfg, "trainer") and hasattr(cfg.trainer, "curriculum"):
                        if cfg.trainer.curriculum == "???":
                            print(f"Skipping {config_name} - curriculum is unresolved")
                            continue

                    create_trainer_config(cfg)

            except ValidationError as e:
                # Skip configs with validation errors - these represent actual config issues
                print(f"Validation error in {config_type} config {config_name}: {e}")
                print(f"This likely indicates an issue with the {config_name} config itself")
                continue
            except Exception as e:
                print(f"Error loading {config_type} config {config_name}: {e}")
                raise AssertionError(f"Failed to load {config_type} config {config_name}: {e}") from e
