"""Example using structured configs for training."""

import logging

from metta.agent import create_agent
from metta.agent.policy_store import MemoryPolicyStore
from metta.rl.functional_trainer import functional_training_loop
from metta.train.train_config import (
    TrainingConfig,
    small_fast_config,
)
from mettagrid import create_env
from mettagrid.curriculum import SimpleCurriculum

logging.basicConfig(level=logging.INFO)


def train_with_config(config: TrainingConfig):
    """Train using a structured config."""

    # 1. Create environment
    env = create_env(
        width=config.environment.width,
        height=config.environment.height,
        max_steps=config.environment.max_steps,
        num_agents=config.environment.num_agents,
        **config.environment.overrides,
    )

    # 2. Create agent
    agent = create_agent(
        agent_name=config.agent.name,
        obs_space=env.observation_space,
        action_space=env.action_space,
        obs_width=env.obs_width,
        obs_height=env.obs_height,
        hidden_size=config.agent.hidden_size,
        device=config.hardware.device,
        **config.agent.params,
    )

    # 3. Create curriculum (if enabled)
    if config.curriculum and config.curriculum.enabled:
        # Use curriculum stages
        curriculum = SimpleCurriculum(stages=config.curriculum.stages)
    else:
        # Simple fixed curriculum
        base_cfg = {
            "width": config.environment.width,
            "height": config.environment.height,
            "max_steps": config.environment.max_steps,
        }
        curriculum = SimpleCurriculum(base_env_config=base_cfg)

    # 4. Create policy store
    policy_store = MemoryPolicyStore()

    # 5. Setup wandb (if enabled)
    wandb_run = None
    if config.wandb and config.wandb.enabled:
        import wandb

        wandb_run = wandb.init(
            project=config.wandb.project,
            name=config.wandb.name or config.experiment_name,
            tags=config.wandb.tags,
            notes=config.wandb.notes or config.notes,
            config=config.to_dict(),
        )

    # 6. Run training
    print(f"Starting training: {config.experiment_name}")
    print(f"Agent: {config.agent.name}, Environment: {config.environment.width}x{config.environment.height}")
    print(f"Total timesteps: {config.trainer.total_timesteps}")

    try:
        state = functional_training_loop(
            config=config.trainer,
            ppo_config=config.ppo,
            policy=agent,
            curriculum=curriculum,
            policy_store=policy_store,
            wandb_run=wandb_run,
        )

        print(f"Training completed! Final epoch: {state.epoch}, Total steps: {state.agent_steps}")

    finally:
        if wandb_run:
            wandb_run.finish()


def main():
    """Example training runs with different configs."""

    # 1. Quick test run
    print("=== Quick Test Run ===")
    test_config = small_fast_config()
    test_config.experiment_name = "quick_test"
    test_config.wandb = None  # Disable wandb for test
    train_with_config(test_config)

    # 2. Custom configuration
    print("\n=== Custom Configuration ===")
    custom_config = TrainingConfig(
        experiment_name="custom_experiment",
        notes="Testing custom configuration with structured configs",
    )

    # Customize individual components
    custom_config.agent.name = "simple_cnn"
    custom_config.agent.hidden_size = 512

    custom_config.environment.width = 20
    custom_config.environment.height = 20
    custom_config.environment.max_steps = 750

    custom_config.trainer.total_timesteps = 50_000
    custom_config.trainer.batch_size = 1024

    custom_config.ppo.clip_coef = 0.3
    custom_config.ppo.ent_coef = 0.02

    custom_config.optimizer.type = "adam"
    custom_config.optimizer.learning_rate = 5e-4

    # Disable wandb for example
    custom_config.wandb = None

    train_with_config(custom_config)

    # 3. Load from existing YAML and convert
    print("\n=== Loading from YAML ===")
    # This shows how to migrate from old YAML configs
    yaml_config = {
        "agent": {"name": "simple_cnn", "hidden_size": 256},
        "environment": {"width": 15, "height": 15},
        "trainer": {"total_timesteps": 100_000, "batch_size": 2048},
        "ppo": {"clip_coef": 0.2},
    }

    loaded_config = TrainingConfig.from_dict(yaml_config)
    loaded_config.experiment_name = "yaml_migration"
    loaded_config.wandb = None

    print(
        f"Loaded config: {loaded_config.agent.name} on {loaded_config.environment.width}x{loaded_config.environment.height}"
    )


if __name__ == "__main__":
    main()
