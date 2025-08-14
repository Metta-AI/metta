"""Training run configuration for tools/train.py."""

from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
import yaml

from metta.common.util.config import Config
from metta.rl.trainer_config import TrainerConfig, OptimizerConfig, PPOConfig


class TrainingRunConfig(Config):
    """Configuration for a training run.

    This contains all the settings that tools/train.py expects,
    including agent, wandb, trainer, etc. This is what gets
    serialized to YAML and sent to the training script.
    """

    # Core training settings
    curriculum: str  # Required - must be explicitly set

    # Agent configuration (references configs/agent/*.yaml)
    agent_config: str = "fast"  # e.g., "fast", "latent_attn_tiny", etc.

    # Simulation configuration (references configs/sim/*.yaml)
    sim_config: str = "arena"  # e.g., "arena", "navigation", etc.

    # WandB settings
    wandb_entity: str = "metta-research"
    wandb_project: str = "metta"
    wandb_tags: Optional[List[str]] = None
    wandb_group: Optional[str] = None
    wandb_notes: Optional[str] = None

    # Training configuration
    trainer: Optional[TrainerConfig] = None

    # Other settings
    seed: int = 1
    py_agent: Optional[str] = None
    bypass_mac_overrides: bool = False
    run_name_pattern: Optional[str] = None

    # Store the path where YAML was saved for local testing
    _saved_yaml_path: Optional[Path] = None

    def get_trainer_config(self) -> TrainerConfig:
        """Get the trainer config object.

        Returns:
            TrainerConfig object with settings
        """
        if self.trainer:
            # Use provided trainer config but override curriculum
            trainer = self.trainer.model_copy()
            trainer.curriculum = self.curriculum
            return trainer
        else:
            # Create default trainer config with minimal settings
            from metta.rl.trainer_config import (
                CheckpointConfig,
                SimulationConfig,
                TorchProfilerConfig,
            )

            return TrainerConfig(
                curriculum=self.curriculum,
                total_timesteps=10_000_000,
                num_workers=4,
                batch_size=2048,
                minibatch_size=512,
                optimizer=OptimizerConfig(
                    type="adam",
                    learning_rate=0.0003,
                ),
                ppo=PPOConfig(
                    clip_coef=0.1,
                    ent_coef=0.01,
                    vf_coef=0.5,
                ),
                checkpoint=CheckpointConfig(
                    checkpoint_dir="${run_dir}/checkpoints",
                ),
                simulation=SimulationConfig(
                    replay_dir="${run_dir}/replays",
                ),
                profiler=TorchProfilerConfig(
                    profile_dir="${run_dir}/torch_traces",
                ),
            )

    def serialize_to_yaml(self) -> Dict[str, Any]:
        """Generate the full training configuration dict.

        Returns:
            Complete config dict matching what tools/train.py expects
        """
        # Use simple defaults - always load standard configs from their normal locations
        config = {
            "defaults": [
                "../common",  # Go up from experiments/ to configs/
                f"../agent/{self.agent_config}",  # Always load agent from configs/agent/
                "../trainer/trainer",
                f"../sim/{self.sim_config}",
                f"../wandb/{'metta_research' if self.wandb_entity == 'metta-research' else 'external_user'}",
                "_self_",
            ],
            "seed": self.seed,
            "py_agent": self.py_agent,
            "train_job": {
                "map_preview_uri": "s3://softmax-public/training_runs/${run}/map_preview.json.z",
                "evals": "${sim}",
            },
            "trainer": self.get_trainer_config().model_dump(exclude_unset=False),
            "bypass_mac_overrides": self.bypass_mac_overrides,
            "run_name_pattern": self.run_name_pattern,
            "cmd": "train",
        }

        # Add wandb overrides if not using default
        if self.wandb_entity != "metta-research":
            config["wandb"] = {
                "entity": self.wandb_entity,
                "project": self.wandb_project,
            }

        if self.wandb_tags:
            config.setdefault("wandb", {})["tags"] = self.wandb_tags

        if self.wandb_group:
            config.setdefault("wandb", {})["group"] = self.wandb_group

        if self.wandb_notes:
            config.setdefault("wandb", {})["notes"] = self.wandb_notes

        return config

    def serialize_to_yaml_file(self, instance_name: str) -> Tuple[Path, Dict[str, Any]]:
        """Serialize the config to a YAML file for transfer.

        Args:
            instance_name: The instance name for the config file (includes timestamp)

        Returns:
            Tuple of (yaml_file_path, full_config_dict)
        """
        # Import here to avoid circular dependency
        from metta.common.util.fs import get_repo_root

        # Save to configs/experiments directory where all the defaults will work
        repo_root = get_repo_root()
        experiments_dir = repo_root / "configs" / "experiments"
        experiments_dir.mkdir(parents=True, exist_ok=True)

        yaml_path = experiments_dir / f"{instance_name}.yaml"

        # Get the full config
        full_config = self.serialize_to_yaml()

        # Write to YAML file with package directive
        with open(yaml_path, "w") as f:
            f.write("# @package _global_\n")
            yaml.dump(full_config, f, default_flow_style=False, sort_keys=False)

        return yaml_path, full_config

    def save_for_local_testing(self, instance_name: str) -> str:
        """Save YAML to the configs directory for local testing.

        Args:
            instance_name: The instance name for the config file (includes timestamp)

        Returns:
            Command to run tools/train.py with this config
        """
        # Use serialize_to_yaml_file to save in the standard location
        yaml_path, _ = self.serialize_to_yaml_file(instance_name=instance_name)
        self._saved_yaml_path = yaml_path

        # Return command to run with the config from experiments/ directory
        yaml_name = yaml_path.stem
        return f"uv run ./tools/train.py +experiments={yaml_name}"
