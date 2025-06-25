from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from pydantic import ValidationError

from metta.rl.trainer_config import MettaTrainerConfig, OptimizerConfig

valid_optimizer_config = {
    "type": "adam",
    "learning_rate": 0.001,
    "beta1": 0.9,
    "beta2": 0.999,
    "eps": 1e-8,
    "weight_decay": 0.0,
}

valid_trainer_config = {
    "_target_": "metta.rl.trainer.MettaTrainer",
    "batch_size": 1024,
    "minibatch_size": 256,
    "total_timesteps": 1000000,
    "optimizer": valid_optimizer_config,
    "clip_coef": 0.1,
    "ent_coef": 0.01,
    "gae_lambda": 0.95,
    "gamma": 0.99,
    "max_grad_norm": 0.5,
    "vf_coef": 0.5,
    "bptt_horizon": 32,
    "update_epochs": 1,
    "forward_pass_minibatch_target_size": 512,
}


class TestTypedConfigs:
    def test_basic_typed_config_parsing(self):
        optimizer = OptimizerConfig.model_validate(valid_optimizer_config)
        assert optimizer.type == "adam"
        assert optimizer.learning_rate == 0.001

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

    def test_trainer_config_with_defaults(self):
        minimal_config = {
            "_target_": "metta.rl.trainer.MettaTrainer",
            "batch_size": 1024,
            "minibatch_size": 256,
            "total_timesteps": 1000000,
            "optimizer": {
                "type": "adam",
                "learning_rate": 0.001,
                # Not fully specified
            },
            "clip_coef": 0.1,
            "ent_coef": 0.01,
            "gae_lambda": 0.95,
            "gamma": 0.99,
            "max_grad_norm": 0.5,
            "vf_coef": 0.5,
            "bptt_horizon": 32,
            "update_epochs": 1,
            "forward_pass_minibatch_target_size": 512,
        }

        trainer = MettaTrainerConfig.model_validate(minimal_config)

        # Defaults
        assert trainer.optimizer.beta1 == 0.9
        assert trainer.optimizer.beta2 == 0.999
        assert trainer.zero_copy is True
        assert trainer.verbose is True

        # Specified
        assert isinstance(trainer.optimizer, OptimizerConfig)
        assert trainer.optimizer.learning_rate == 0.001


configs_dir = str(Path(__file__).parent.parent.parent / "configs" / "trainer")


def load_config_with_hydra(trainer_name: str, overrides: list[str] | None = None) -> DictConfig:
    configs_dir = str(Path(__file__).parent.parent.parent / "configs")
    default_overrides = [
        "run=test_run",  # Set run first so run_dir interpolation works
        f"trainer={trainer_name}",
        "wandb=off",  # Disable wandb for tests
    ]

    with initialize_config_dir(config_dir=configs_dir, version_base=None):
        return compose(
            config_name="train_job",
            overrides=default_overrides + (overrides or []),
        )


class TestRealTypedConfigs:
    def test_loaded_g16_typed_config_equivalent_to_dictconfig(self):
        # This is how @hydra.main would load the config
        cfg = load_config_with_hydra("g16", ["trainer.num_workers=1"])

        trainer_cfg = cfg.trainer  # Access trainer config directly

        # Test three ideally-equivalent ways to work with the config
        configs: list[MettaTrainerConfig | DictConfig] = [
            MettaTrainerConfig.model_validate(OmegaConf.to_container(trainer_cfg, resolve=True)),
            MettaTrainerConfig.model_validate(trainer_cfg),
            trainer_cfg,
        ]

        # Verify all three approaches give equivalent results
        for config in configs:
            # manual overrides
            assert config.num_workers == 1

            # g16 overrides
            assert config.batch_size == 524288  # From g16.yaml
            assert config.update_epochs == 2  # From g16.yaml
            assert config.optimizer.learning_rate == 0.0005570885525880004  # From g16.yaml

            # inherited values
            assert config.total_timesteps == 50_000_000_000  # From puffer.yaml
            assert config.optimizer.type == "adam"  # From puffer.yaml
            assert config.cpu_offload is False  # From puffer.yaml

    def test_all_trainer_configs_comprehensive(self):
        config_files = [f.stem for f in Path(configs_dir).glob("*.yaml")]

        for config_name in config_files:
            # Skip trainer.yaml; it isn't a full config
            if config_name == "trainer":
                continue

            try:
                # some of the configs don't have num_workers specified, leaning instead on
                # train.sh to provide it as a command-line argument.
                cfg = load_config_with_hydra(config_name, overrides=["trainer.num_workers=1"])

                validated_config = MettaTrainerConfig.model_validate(cfg.trainer)

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
