"""
Arena Basic Easy Shaped Rewards Recipe - Pre-dehydration compatible version

This recipe exactly recreates the pre-dehydration default training configuration:
- Matches the old ./tools/train.py default which used basic_easy_shaped environment
- Includes all fixes discovered during performance investigation
- Should achieve ~20 heart.get as before
"""

from typing import List, Optional

import metta.cogworks.curriculum as cc
import metta.map.scenes.random
import metta.mettagrid.config.envs as eb
from metta.map.mapgen import MapGen
from metta.mettagrid.config import building
from metta.mettagrid.mettagrid_config import (
    ActionsConfig,
    ActionConfig,
    AttackActionConfig,
    ChangeGlyphActionConfig,
    EnvConfig,
)
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
    """Create the pre-dehydration basic_easy_shaped environment exactly."""

    # Start with standard arena configuration
    env_cfg = eb.make_arena(num_agents=num_agents, combat=False)

    # CRITICAL: Recreate the exact OLD map configuration from configs/env/mettagrid/arena/basic.yaml
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
                    "block": 20,  # CRITICAL: Blocks were in old config
                    "wall": 20,  # CRITICAL: 20 walls, not 10
                },
            ),
        ),
    )

    # Add block object (was in old basic.yaml)
    env_cfg.game.objects["block"] = building.block

    # Remove combat buildings not in old basic config
    if "lasery" in env_cfg.game.objects:
        del env_cfg.game.objects["lasery"]
    if "armory" in env_cfg.game.objects:
        del env_cfg.game.objects["armory"]

    # ACTION SET: Using move_8way since move no longer exists
    # The old config had move+rotate, but now we need to use move_8way which combines both
    env_cfg.game.actions = ActionsConfig(
        noop=ActionConfig(enabled=True),
        move_8way=ActionConfig(
            enabled=True
        ),  # 8-directional movement (replaces old move+rotate)
        rotate=ActionConfig(enabled=True),  # Keep rotation for compatibility
        put_items=ActionConfig(enabled=True),
        get_items=ActionConfig(enabled=True),
        attack=AttackActionConfig(
            enabled=True,
            consumed_resources={"laser": 1},
            defense_resources={"armor": 1},
        ),
        swap=ActionConfig(enabled=True),
        # These were disabled in old config
        place_box=ActionConfig(enabled=False),
        change_color=ActionConfig(enabled=False),
        change_glyph=ChangeGlyphActionConfig(enabled=False, number_of_glyphs=4),
    )

    # Shaped rewards exactly matching old configs/env/mettagrid/game/agent/rewards/shaped.yaml
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

    # CRITICAL FIX: heart_max was null (unlimited) in old config
    # In new system, 255 is the maximum possible value
    env_cfg.game.agent.rewards.inventory.heart = 1
    env_cfg.game.agent.rewards.inventory.heart_max = (
        255  # Was 100, now maximum possible
    )

    # Easy converter configuration (from configs/env/mettagrid/game/objects/basic_easy.yaml)
    # Altar only needs 1 battery_red instead of 3
    altar_copy = env_cfg.game.objects["altar"].model_copy(deep=True)
    altar_copy.input_resources = {"battery_red": 1}
    env_cfg.game.objects["altar"] = altar_copy

    # CRITICAL FIX: Set initial resource counts so buildings aren't empty!
    # Without this, agents have to wait for production cycles before getting any rewards
    for obj_name in ["mine_red", "generator_red", "altar"]:
        if obj_name in env_cfg.game.objects:
            obj_copy = env_cfg.game.objects[obj_name].model_copy(deep=True)
            obj_copy.initial_resource_count = 1  # Start with 1 resource ready
            env_cfg.game.objects[obj_name] = obj_copy

    # Set label for clarity
    env_cfg.label = "arena.basic_easy_shaped"

    # Global configuration flags from old mettagrid.yaml
    env_cfg.desync_episodes = True  # Explicit: changes max_steps for first episode only
    env_cfg.game.track_movement_metrics = True
    env_cfg.game.no_agent_interference = False
    env_cfg.game.resource_loss_prob = 0.0
    env_cfg.game.recipe_details_obs = False

    # Global observation tokens from old config
    env_cfg.game.global_obs.episode_completion_pct = True
    env_cfg.game.global_obs.last_action = True
    env_cfg.game.global_obs.last_reward = True
    env_cfg.game.global_obs.resource_rewards = False
    env_cfg.game.global_obs.visitation_counts = False

    return env_cfg


def make_evals(env: Optional[EnvConfig] = None) -> List[SimulationConfig]:
    """Create evaluation environments."""
    basic_env = env or make_env()

    # Basic evaluation without combat
    basic_env.game.actions.attack.consumed_resources["laser"] = 100

    # Combat evaluation with attacks enabled
    combat_env = basic_env.model_copy()
    combat_env.game.actions.attack.consumed_resources["laser"] = 1

    return [
        SimulationConfig(name="arena_basic_easy_shaped/basic", env=basic_env),
        SimulationConfig(name="arena_basic_easy_shaped/combat", env=combat_env),
    ]


def train() -> TrainTool:
    """
    Create training configuration matching pre-dehydration defaults.

    This matches:
    - configs/train_job.yaml default settings
    - configs/trainer/trainer.yaml hyperparameters
    - basic_easy_shaped environment configuration
    """
    env_cfg = make_env()

    # Log configuration for verification
    print("\n[Arena Basic Easy Shaped - Pre-dehydration Compatible]")
    print("Environment configuration:")
    print(
        f"  - Altar: requires {env_cfg.game.objects['altar'].input_resources['battery_red']} battery (easy mode)"
    )
    print(
        f"  - Initial resources: {env_cfg.game.objects['altar'].initial_resource_count}"
    )
    print(
        f"  - Mine initial: {env_cfg.game.objects['mine_red'].initial_resource_count}"
    )
    print(
        f"  - Generator initial: {env_cfg.game.objects['generator_red'].initial_resource_count}"
    )
    print("  - Shaped rewards:")
    print(
        f"    * ore_red: {env_cfg.game.agent.rewards.inventory.ore_red} (max: {env_cfg.game.agent.rewards.inventory.ore_red_max})"
    )
    print(
        f"    * battery_red: {env_cfg.game.agent.rewards.inventory.battery_red} (max: {env_cfg.game.agent.rewards.inventory.battery_red_max})"
    )
    print(
        f"    * heart: {env_cfg.game.agent.rewards.inventory.heart} (max: {env_cfg.game.agent.rewards.inventory.heart_max})"
    )
    print(f"  - Map has blocks: {'block' in env_cfg.game.objects}")
    print(f"  - Num agents: {env_cfg.game.num_agents}")
    print(f"  - Max steps: {env_cfg.game.max_steps}")
    print(
        "  - Actions enabled: move_8way, rotate, get_items, put_items, attack, swap, noop"
    )
    print("\nKey fixes applied:")
    print("  ✓ initial_resource_count = 1 (immediate rewards)")
    print("  ✓ heart_max = 255 (was capped at 100)")
    print("  ✓ Minimal action set (easier learning)")
    print("  ✓ Blocks and walls in map")

    # Create trainer configuration with original hyperparameters
    trainer_cfg = TrainerConfig(
        # Environment
        curriculum=cc.env_curriculum(env_cfg),
        # Training duration - default was 10B
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


def play(
    env: Optional[EnvConfig] = None,
    num_agents: int = 24,
    policy_uri: Optional[str] = None,
) -> PlayTool:
    """Interactive play tool for testing the environment."""
    if env is None:
        print("\n=== Arena Basic Easy Shaped (Pre-dehydration Compatible) ===")
        print(f"Creating environment with {num_agents} agents")
        print("Key features:")
        print("  - Easy altar (1 battery → 1 heart)")
        print("  - Initial resources available immediately")
        print("  - Heart rewards uncapped (max 255)")
        print("  - Minimal action set for easier learning")
        print("  - Shaped rewards for progression")
        print("")

        if policy_uri:
            print(f"WATCHING TRAINED POLICY: {policy_uri}")
            print("The agents will act autonomously using the trained policy.")
        else:
            print("MANUAL CONTROL MODE")
            print("Controls:")
            print("  - Click on an agent to select it")
            print("  - Use arrow keys for cardinal movement")
            print("  - Auto-interact when facing objects")
            print("")
            print("Gameplay:")
            print("  1. Mine red ore from mines (reward: 0.1 per ore, max 1.0)")
            print(
                "  2. Convert ore to batteries at generators (reward: 0.8 per battery, max 1.0)"
            )
            print(
                "  3. Convert batteries to hearts at altars (reward: 1.0 per heart, max 255.0)"
            )
        print("")

        eval_env = make_env(num_agents=num_agents)
    else:
        eval_env = env

    return PlayTool(
        sim=SimulationConfig(env=eval_env, name="arena_basic_easy_shaped"),
        policy_uri=policy_uri,
    )


def replay(env: Optional[EnvConfig] = None) -> ReplayTool:
    """Replay tool for viewing recorded episodes."""
    eval_env = env or make_env()
    return ReplayTool(
        sim=SimulationConfig(env=eval_env, name="arena_basic_easy_shaped")
    )


def evaluate(policy_uri: str) -> SimTool:
    """Evaluate a trained policy on arena environments."""
    return SimTool(
        simulations=make_evals(),
        policy_uris=[policy_uri],
    )


if __name__ == "__main__":
    """Allow running this recipe directly for play testing.
    
    Usage:
        uv run experiments/recipes/arena_basic_easy_shaped.py [port] [num_agents_or_policy] [policy]
        
    Examples:
        uv run experiments/recipes/arena_basic_easy_shaped.py              # Default: port 8001, 24 agents
        uv run experiments/recipes/arena_basic_easy_shaped.py 8002 6        # Port 8002, 6 agents
        uv run experiments/recipes/arena_basic_easy_shaped.py 8003 wandb://run/my-run
    """
    import os
    import sys

    # Parse arguments
    port = 8001
    num_agents = 24
    policy_uri = None

    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"Invalid port: {sys.argv[1]}, using default {port}")

    if len(sys.argv) > 2:
        arg = sys.argv[2]
        if arg.startswith("wandb://") or arg.startswith("file://") or "/" in arg:
            policy_uri = arg
        else:
            try:
                num_agents = int(arg)
                if num_agents % 6 != 0:
                    num_agents = (num_agents // 6) * 6
            except ValueError:
                policy_uri = arg

    if len(sys.argv) > 3 and not policy_uri:
        policy_uri = sys.argv[3]

    # Set server port
    os.environ["METTASCOPE_PORT"] = str(port)

    # Run play
    play_tool = play(num_agents=num_agents, policy_uri=policy_uri)
    play_tool.invoke({}, [])
