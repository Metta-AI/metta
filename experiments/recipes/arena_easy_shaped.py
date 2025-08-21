"""
Arena Easy Shaped Rewards Recipe

This recipe recreates the basic_easy_shaped configuration from the old YAML system.
It includes:
- Arena environment with 24 agents
- Shaped rewards for ore, batteries, lasers, armor, and blueprints
- Easy converter configuration (altar only needs 1 battery_red)
- Standard training hyperparameters from the original config
"""

from typing import List, Optional

import metta.cogworks.curriculum as cc
import metta.map.scenes.random
import metta.mettagrid.config.envs as eb
from metta.map.mapgen import MapGen
from metta.mettagrid.config import building
from metta.mettagrid.mettagrid_config import EnvConfig
from metta.rl.trainer_config import (
    CheckpointConfig,
    EvaluationConfig,
    OptimizerConfig,
    PPOConfig,
    TrainerConfig,
)
from metta.sim.simulation_config import SimulationConfig
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
from metta.tools.sim import SimTool
from metta.tools.train import TrainTool


def make_env(num_agents: int = 24) -> EnvConfig:
    """Create the arena easy shaped rewards environment matching the old basic_easy_shaped."""

    # Start with the standard arena configuration but we'll override the map
    env_cfg = eb.make_arena(num_agents=num_agents, combat=False)

    # CRITICAL: Recreate the OLD map configuration with blocks and more walls
    # This matches configs/env/mettagrid/arena/basic.yaml exactly
    env_cfg.game.map_builder = MapGen.Config(
        num_agents=num_agents,
        width=25,
        height=25,
        border_width=6,
        instance_border_width=0,
        root=metta.map.scenes.random.Random.factory(
            params=metta.map.scenes.random.Random.Params(
                agents=6,
                objects={
                    "mine_red": 10,
                    "generator_red": 5,
                    "altar": 5,
                    "block": 20,  # CRITICAL: Add blocks back!
                    "wall": 20,  # CRITICAL: 20 walls not 10!
                },
            ),
        ),
    )

    # Add block object to the environment (was missing in new version)
    env_cfg.game.objects["block"] = building.block

    # Remove combat buildings that weren't in the old config
    if "lasery" in env_cfg.game.objects:
        del env_cfg.game.objects["lasery"]
    if "armory" in env_cfg.game.objects:
        del env_cfg.game.objects["armory"]

    # Set shaped rewards (from configs/env/mettagrid/game/agent/rewards/shaped.yaml)
    env_cfg.game.agent.rewards.inventory.ore_red = 0.1
    env_cfg.game.agent.rewards.inventory.ore_red_max = 1

    env_cfg.game.agent.rewards.inventory.battery_red = 0.8
    env_cfg.game.agent.rewards.inventory.battery_red_max = 1

    env_cfg.game.agent.rewards.inventory.laser = 0.5
    env_cfg.game.agent.rewards.inventory.laser_max = 1

    env_cfg.game.agent.rewards.inventory.armor = 0.5
    env_cfg.game.agent.rewards.inventory.armor_max = 1

    env_cfg.game.agent.rewards.inventory.blueprint = 0.5
    env_cfg.game.agent.rewards.inventory.blueprint_max = 1

    env_cfg.game.agent.rewards.inventory.heart = 1
    env_cfg.game.agent.rewards.inventory.heart_max = 100

    # Easy converter configuration (from configs/env/mettagrid/game/objects/basic_easy.yaml)
    # Altar only needs 1 battery_red instead of the default (harder) requirement
    # Need to make a copy since it's a pydantic model
    altar_copy = env_cfg.game.objects["altar"].model_copy(deep=True)
    altar_copy.input_resources = {"battery_red": 1}
    env_cfg.game.objects["altar"] = altar_copy

    # Set label for clarity
    env_cfg.label = "arena.easy_shaped"

    # desync_episodes is now fixed to match old behavior (changes max_steps for first episode only)
    # Keep default setting (True) to match pre-dehydration behavior

    # Log a warning about potential CurriculumEnv issues
    print("\nWARNING: CurriculumEnv wrapper may be affecting performance.")
    print("It calls set_env_config after every episode which rebuilds map_builder.")
    print("Consider testing without the wrapper if performance issues persist.\n")

    return env_cfg


def make_evals(env: Optional[EnvConfig] = None) -> List[SimulationConfig]:
    """Create evaluation environments for arena easy shaped."""
    basic_env = env or make_env()

    # Basic evaluation without combat
    basic_env.game.actions.attack.consumed_resources["laser"] = 100

    # Combat evaluation with attacks enabled
    combat_env = basic_env.model_copy()
    combat_env.game.actions.attack.consumed_resources["laser"] = 1

    return [
        SimulationConfig(name="arena_easy_shaped/basic", env=basic_env),
        SimulationConfig(name="arena_easy_shaped/combat", env=combat_env),
    ]


def train() -> TrainTool:
    """
    Create training configuration for arena easy shaped rewards.

    This matches the original train_job.yaml configuration:
    - Uses the basic_easy_shaped environment
    - Standard hyperparameters from configs/trainer/trainer.yaml
    - Total timesteps: 10B (can be overridden at runtime)
    """
    env_cfg = make_env()

    # Log environment configuration for verification
    print(f"\n[Arena Easy Shaped] Creating training with environment config:")
    print(
        f"  - Altar: input={env_cfg.game.objects['altar'].input_resources}, "
        f"cooldown={env_cfg.game.objects['altar'].cooldown}"
    )
    print(f"  - Mine: cooldown={env_cfg.game.objects['mine_red'].cooldown}")
    print(f"  - Generator: cooldown={env_cfg.game.objects['generator_red'].cooldown}")
    print(f"  - Shaped rewards:")
    print(
        f"    * ore_red: {env_cfg.game.agent.rewards.inventory.ore_red} (max: {env_cfg.game.agent.rewards.inventory.ore_red_max})"
    )
    print(
        f"    * battery_red: {env_cfg.game.agent.rewards.inventory.battery_red} (max: {env_cfg.game.agent.rewards.inventory.battery_red_max})"
    )
    print(
        f"    * heart: {env_cfg.game.agent.rewards.inventory.heart} (max: {env_cfg.game.agent.rewards.inventory.heart_max})"
    )
    print(f"  - Map objects: {list(env_cfg.game.objects.keys())}")
    print(f"  - Has blocks: {'block' in env_cfg.game.objects}")
    print(f"  - Max steps: {env_cfg.game.max_steps}")
    print(f"  - Num agents: {env_cfg.game.num_agents}")

    # Create trainer configuration with original hyperparameters
    trainer_cfg = TrainerConfig(
        # Environment
        curriculum=cc.env_curriculum(env_cfg),
        # Training duration
        total_timesteps=10_000_000_000,
        # Checkpointing (from original config)
        checkpoint=CheckpointConfig(
            checkpoint_interval=50,
            wandb_checkpoint_interval=50,
        ),
        # Evaluation
        evaluation=EvaluationConfig(
            simulations=make_evals(env_cfg),
            evaluate_interval=50,
            evaluate_remote=True,
            evaluate_local=False,
        ),
        # Optimizer settings (from original trainer.yaml)
        optimizer=OptimizerConfig(
            type="adam",
            learning_rate=0.000457,
            beta1=0.9,
            beta2=0.999,
            eps=1e-12,
            weight_decay=0,
        ),
        # PPO hyperparameters (from original trainer.yaml)
        ppo=PPOConfig(
            clip_coef=0.1,
            ent_coef=0.0021,
            gae_lambda=0.916,
            gamma=0.977,
            max_grad_norm=0.5,
            vf_clip_coef=0.1,
            vf_coef=0.44,
            l2_reg_loss_coef=0,
            l2_init_loss_coef=0,
            norm_adv=True,
            clip_vloss=True,
            target_kl=None,
        ),
        # Batch configuration (from original trainer.yaml)
        batch_size=524288,
        minibatch_size=16384,
        bptt_horizon=64,
        update_epochs=1,
        # Performance settings
        zero_copy=True,
        require_contiguous_env_ids=False,
        verbose=True,
        cpu_offload=False,
        compile=False,
        compile_mode="reduce-overhead",
        forward_pass_minibatch_target_size=4096,
        async_factor=2,
        scale_batches_by_world_size=False,
    )

    return TrainTool(trainer=trainer_cfg)


def play(env: Optional[EnvConfig] = None, num_agents: int = 24) -> PlayTool:
    """Interactive play tool for testing the environment.
    
    Args:
        env: Optional environment config to use (defaults to make_env())
        num_agents: Number of agents (default 24 for full arena, can use 6 for simpler testing)
    """
    if env is None:
        # Create the full arena easy shaped environment
        print("\n=== Arena Easy Shaped Environment ===")
        print(f"Creating environment with:")
        print(f"  - {num_agents} agents (cooperative)")
        print(f"  - Red mines (produce ore)")
        print(f"  - Red generators (convert ore to batteries)")
        print(f"  - Altars (convert 1 battery to hearts - EASY MODE)")
        print(f"  - Blocks and walls for obstacles")
        print(f"  - Shaped rewards for progression")
        print("=====================================\n")
        
        eval_env = make_env(num_agents=num_agents)
        
        # Print gameplay instructions
        print("Controls:")
        print("  - Click on an agent to select it")
        print("  - Use arrow keys or WASD to move")
        print("  - Press spacebar to interact with objects")
        print("")
        print("Gameplay:")
        print("  1. Mine red ore from mines (reward: 0.1 per ore, max 1.0)")
        print("  2. Convert ore to batteries at generators (reward: 0.8 per battery, max 1.0)")
        print("  3. Convert batteries to hearts at altars (reward: 1.0 per heart, max 100.0)")
        print("")
    else:
        eval_env = env
    
    return PlayTool(sim=SimulationConfig(env=eval_env, name="arena_easy_shaped"))


def replay(env: Optional[EnvConfig] = None) -> ReplayTool:
    """Replay tool for viewing recorded episodes."""
    eval_env = env or make_env()
    return ReplayTool(sim=SimulationConfig(env=eval_env, name="arena_easy_shaped"))


def evaluate(policy_uri: str) -> SimTool:
    """Evaluate a trained policy on arena easy shaped environments."""
    return SimTool(
        simulations=make_evals(),
        policy_uris=[policy_uri],
    )
