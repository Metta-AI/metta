"""Migration utilities for converting from Hydra to library architecture."""

from typing import Any, Dict, Optional

import yaml
from omegaconf import DictConfig, OmegaConf

from metta.agent import list_agents
from metta.train.config import AgentConfig, OptimizerConfig, TrainerConfig, TrainingConfig


def hydra_config_to_training_config(hydra_cfg: DictConfig) -> TrainingConfig:
    """Convert a Hydra configuration to TrainingConfig.

    Args:
        hydra_cfg: Hydra DictConfig object

    Returns:
        TrainingConfig instance
    """
    # Extract agent name from Hydra config
    agent_name = "simple_cnn"  # default
    if hasattr(hydra_cfg, "agent"):
        if hasattr(hydra_cfg.agent, "name"):
            agent_name = hydra_cfg.agent.name
        elif hasattr(hydra_cfg.agent, "_target_"):
            # Try to infer from target
            target = hydra_cfg.agent._target_
            if "simple" in target.lower():
                agent_name = "simple_cnn"
            elif "large" in target.lower():
                agent_name = "large_cnn"
            elif "attention" in target.lower():
                agent_name = "attention"

    # Build optimizer config
    optimizer_cfg = OptimizerConfig()
    if hasattr(hydra_cfg, "trainer") and hasattr(hydra_cfg.trainer, "optimizer"):
        opt = hydra_cfg.trainer.optimizer
        optimizer_cfg = OptimizerConfig(
            type=getattr(opt, "type", "adam"),
            learning_rate=getattr(opt, "learning_rate", 3e-4),
            beta1=getattr(opt, "beta1", 0.9),
            beta2=getattr(opt, "beta2", 0.999),
            eps=getattr(opt, "eps", 1e-8),
            weight_decay=getattr(opt, "weight_decay", 0.0),
        )

    # Build trainer config
    trainer_cfg = TrainerConfig(optimizer=optimizer_cfg)
    if hasattr(hydra_cfg, "trainer"):
        t = hydra_cfg.trainer
        trainer_cfg = TrainerConfig(
            total_timesteps=getattr(t, "total_timesteps", 10_000_000),
            batch_size=getattr(t, "batch_size", 32768),
            minibatch_size=getattr(t, "minibatch_size", 2048),
            num_workers=getattr(t, "num_workers", 1),
            checkpoint_dir=getattr(t, "checkpoint_dir", "./checkpoints"),
            checkpoint_interval=getattr(t, "checkpoint_interval", 60),
            wandb_checkpoint_interval=getattr(t, "wandb_checkpoint_interval", 300),
            evaluate_interval=getattr(t, "evaluate_interval", 300),
            replay_interval=getattr(t, "replay_interval", 300),
            device=getattr(t, "device", hydra_cfg.get("device", "cuda")),
            compile=getattr(t, "compile", False),
            compile_mode=getattr(t, "compile_mode", "default"),
            optimizer=optimizer_cfg,
        )

    # Build agent config
    agent_cfg = AgentConfig(name=agent_name)
    if hasattr(hydra_cfg, "agent"):
        a = hydra_cfg.agent
        # Extract hidden size and lstm layers from components if available
        hidden_size = 128
        lstm_layers = 2

        if hasattr(a, "components") and hasattr(a.components, "_core_"):
            hidden_size = getattr(a.components._core_, "output_size", 128)
            if hasattr(a.components._core_, "nn_params"):
                lstm_layers = getattr(a.components._core_.nn_params, "num_layers", 2)

        agent_cfg = AgentConfig(
            name=agent_name,
            hidden_size=hidden_size,
            lstm_layers=lstm_layers,
            clip_range=getattr(a, "clip_range", 0.0),
            analyze_weights_interval=getattr(a, "analyze_weights_interval", 300),
            l2_norm_coeff=getattr(a, "l2_norm_coeff", 0.0),
            l2_init_coeff=getattr(a, "l2_init_coeff", 0.0),
            l2_init_weight_update_interval=getattr(a, "l2_init_weight_update_interval", 0),
        )

    # Build main config
    return TrainingConfig(
        run_name=hydra_cfg.get("run", "metta_run"),
        run_dir=hydra_cfg.get("run_dir", "./runs/metta_run"),
        data_dir=hydra_cfg.get("data_dir", "./data"),
        trainer=trainer_cfg,
        agent=agent_cfg,
        vectorization=hydra_cfg.get("vectorization", 1024),
    )


def convert_yaml_config(yaml_path: str, output_path: Optional[str] = None) -> TrainingConfig:
    """Convert a YAML configuration file to the new format.

    Args:
        yaml_path: Path to the Hydra YAML config
        output_path: Optional path to save the converted config

    Returns:
        TrainingConfig instance
    """
    with open(yaml_path, "r") as f:
        yaml_data = yaml.safe_load(f)

    hydra_cfg = OmegaConf.create(yaml_data)
    training_cfg = hydra_config_to_training_config(hydra_cfg)

    if output_path:
        training_cfg.to_yaml(output_path)
        print(f"Converted config saved to: {output_path}")

    return training_cfg


def suggest_agent_migration(hydra_agent_config: Dict[str, Any]) -> str:
    """Suggest how to migrate a Hydra agent config to the new architecture.

    Args:
        hydra_agent_config: Hydra agent configuration dict

    Returns:
        Suggested Python code for the agent
    """
    components = hydra_agent_config.get("components", {})

    # Analyze the architecture
    has_attention = any("attention" in name.lower() for name in components.keys())
    num_cnn_layers = sum(1 for name in components.keys() if "cnn" in name.lower())

    # Suggest an agent
    if has_attention:
        if "cross" in str(components).lower():
            suggested_agent = "multi_head_attention"
        else:
            suggested_agent = "attention"
    elif num_cnn_layers >= 3:
        suggested_agent = "large_cnn"
    else:
        suggested_agent = "simple_cnn"

    # Generate code suggestion
    code = f"""
# Suggested migration to new architecture:

from metta.agent import {suggested_agent.replace("_", " ").title().replace(" ", "")}Agent

agent = create_agent(
    agent_name="{suggested_agent}",
    obs_space=env.observation_space,
    action_space=env.action_space,
    obs_width=env.obs_width,
    obs_height=env.obs_height,
    feature_normalizations=env.feature_normalizations,
    device="cuda",
)

# Available agents: {", ".join(list_agents())}
"""

    return code


def validate_migration(old_config_path: str, new_config: TrainingConfig) -> bool:
    """Validate that a migration preserved important settings.

    Args:
        old_config_path: Path to original Hydra config
        new_config: Migrated TrainingConfig

    Returns:
        True if migration looks valid
    """
    with open(old_config_path, "r") as f:
        old_data = yaml.safe_load(f)

    issues = []

    # Check critical parameters
    if "trainer" in old_data:
        old_trainer = old_data["trainer"]
        if "total_timesteps" in old_trainer:
            if old_trainer["total_timesteps"] != new_config.trainer.total_timesteps:
                issues.append(
                    f"total_timesteps mismatch: {old_trainer['total_timesteps']} vs {new_config.trainer.total_timesteps}"
                )

    if issues:
        print("Migration validation issues:")
        for issue in issues:
            print(f"  - {issue}")
        return False

    return True
