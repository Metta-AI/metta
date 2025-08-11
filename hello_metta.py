#!/usr/bin/env -S uv run

"""Hello Metta demo with curriculum - demonstrates how to train with a bucketed curriculum."""

from datetime import datetime
from pathlib import Path

from cogworks.curriculum.builders import CurriculumBuilder, TaskSetBuilder
from cogworks.util.tool import run_tool
from metta.mettagrid.config import object as objects
from metta.mettagrid.mettagrid_config import (
    ActionConfig,
    ActionsConfig,
    AgentConfig,
    AgentRewards,
    EnvConfig,
    GameConfig,
    GroupConfig,
    InventoryRewards,
    MapConfig,
)
from metta.rl.trainer_config import TrainerConfig
from tools.train import TrainConfig


def main():
    """Main demo function with curriculum learning."""
    print("🎯 Hello Metta Demo - Curriculum Learning with Map Width Buckets")
    print("=" * 65)

    print("📝 Creating curriculum with map width bucketed tasks...")
    print("   - Tasks bucketed by map_width: [5, 10]")
    print("   - 5 agents per environment")
    print("   - 1 altar (converts battery_red → heart)")
    print("   - Resource chain: mine_red → ore_red → generator_red → battery_red → heart")
    print()

    # Create base environment config
    base_env_config = EnvConfig(
        game=GameConfig(
            num_agents=5,
            max_steps=1000,
            inventory_item_names=["ore_red", "battery_red", "heart"],
            # Agent configuration with simple rewards
            agent=AgentConfig(
                default_resource_limit=50,
                resource_limits={"heart": 100, "battery_red": 50},
                rewards=AgentRewards(inventory=InventoryRewards(ore_red=0.1, battery_red=0.2, heart=1.0)),
            ),
            groups={"agent": GroupConfig(id=0, sprite=0, props=AgentConfig())},
            # Simple actions
            actions=ActionsConfig(
                noop=ActionConfig(),
                move=ActionConfig(),
                get_items=ActionConfig(),
                put_items=ActionConfig(),
            ),
            # Objects: Use predefined objects from objects.py
            objects={
                "altar": objects.altar,
                "mine_red": objects.mine_red,
                "generator_red": objects.generator_red,
            },
            # Base map configuration - will be varied by curriculum
            map=MapConfig(
                width=10,  # Default, will be overridden by curriculum buckets
                height=10,  # Will match width to keep maps square
                border_width=2,
                seed=42,
                root={
                    "type": "Random",
                    "params": {
                        "width": 10,
                        "height": 10,
                        "objects": {"altar": 1, "mine_red": 2, "generator_red": 1},
                        "agents": 5,
                        "border_width": 2,
                    },
                },
            ),
        ),
        desync_episodes=True,
    )

    # Create bucketed task set with map width variations
    bucketed_task_set = (
        TaskSetBuilder.bucketed(base_env_config)
        .add_value_bucket("game.map.width", [5, 10])
        .add_value_bucket("game.map.height", [5, 10])  # Keep maps square
        .add_value_bucket("game.map.root.params.width", [5, 10])
        .add_value_bucket("game.map.root.params.height", [5, 10])
        .add_value_bucket("game.map.root.params.border_width", [1, 2])  # Adjust border for smaller maps
        .build()
    )

    # Create random curriculum from the bucketed task set
    curriculum = CurriculumBuilder.random(bucketed_task_set).build()

    print("✅ Curriculum created with:")
    print("   - Map width/height buckets: [5x5, 10x10]")
    print("   - Border width buckets: [1, 2]")
    print("   - Random task selection from buckets")
    print()

    # Create a TrainerConfig with curriculum
    run_id = f"hello_metta_curriculum_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    train_dir = f"./train_dir/{run_id}"

    trainer_config = TrainerConfig(
        total_timesteps=100000,  # More steps to see curriculum learning
        num_workers=2,  # Use more workers for curriculum training
        curriculum=curriculum,  # Use curriculum instead of single env
        # PPO settings - using defaults but with small batch size
        batch_size=1024,
        minibatch_size=128,
        # Checkpoint settings - must provide valid paths
        checkpoint={
            "checkpoint_interval": 20000,
            "checkpoint_dir": f"{train_dir}/checkpoints",
            "wandb_checkpoint_interval": 0,
        },
        # Simulation settings - must provide valid replay_dir
        simulation={
            "evaluate_interval": 0,
            "replay_dir": f"{train_dir}/replays",
            "evaluate_remote": False,
            "evaluate_local": False,
        },
    )

    # Create the main train config
    config = TrainConfig(
        run=run_id,
        trainer=trainer_config,
        device="cpu",
        wandb={"enabled": False},  # Disable wandb for demo
    )

    print("🚀 Training Configuration:")
    print(f"   Run ID: {run_id}")
    print(f"   Total timesteps: {config.trainer.total_timesteps:,}")
    print(f"   Device: {config.device}")
    print(f"   Num workers: {config.trainer.num_workers}")
    print("   Using curriculum with bucketed map sizes")
    print()

    # Create train_dir structure
    train_dir = Path("train_dir") / run_id
    train_dir.mkdir(parents=True, exist_ok=True)

    print("🎬 Starting curriculum training...")
    print("-" * 40)

    # Use run_tool to save config and run training
    run_tool("train", config, train_dir)
    print("✅ Curriculum training completed!")

    print()
    print("📁 Training outputs:")
    if train_dir.exists():
        for item in train_dir.rglob("*"):
            if item.is_file():
                print(f"   {item}")

    print()
    print("🎉 Curriculum Demo Completed!")
    print("   This demo shows how to train with a bucketed curriculum:")
    print("   - Tasks varied by map width: [5, 10]")
    print("   - Random selection from curriculum buckets")
    print("   - 5 agents per environment learning resource collection")
    print(f"   Config file: {train_dir}/tool_config.yaml")
    print()
    print("🎮 Game Rules:")
    print("   - Mine red ore from mines")
    print("   - Convert ore to batteries at generators")
    print("   - Use altar to convert 3 batteries → 1 heart")
    print("   - Agents get rewards for collecting hearts!")
    print("   - Training across different map sizes for generalization")


if __name__ == "__main__":
    main()
