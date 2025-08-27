"""
Arena Basic Easy Shaped Rewards Recipe for Transformer Agent Testing

This recipe adapts arena_basic_easy_shaped for testing the improved transformer agent.
It uses the same environment configuration but is specifically designed for testing
transformer architectures with enhanced capacity and GTrXL features.
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
from metta.agent.agent_config import AgentConfig
from metta.rl.trainer_config import (
    CheckpointConfig,
    EvaluationConfig,
    TrainerConfig,
)
from metta.sim.simulation_config import SimulationConfig
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
from metta.tools.sim import SimTool
from metta.tools.train import TrainTool


def make_env(num_agents: int = 24) -> EnvConfig:
    """Create environment optimized for transformer agent testing."""

    # Start with standard arena configuration
    env_cfg = eb.make_arena(num_agents=num_agents, combat=False)

    # Map configuration optimized for transformer learning
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
                    "mine_red": 12,  # Slightly more mines for richer observations
                    "generator_red": 6,  # More generators for complex interactions
                    "altar": 6,  # More altars for goal variety
                    "block": 25,  # More blocks for complex navigation
                    "wall": 25,  # More walls for spatial reasoning
                },
            ),
        ),
    )

    # Add block object
    env_cfg.game.objects["block"] = building.block

    # Action configuration using move_8way for directional movement
    env_cfg.game.actions = ActionsConfig(
        noop=ActionConfig(enabled=True),
        move_8way=ActionConfig(enabled=True),  # 8-directional movement
        rotate=ActionConfig(enabled=True),  # Rotation action
        put_items=ActionConfig(enabled=True),
        get_items=ActionConfig(enabled=True),
        attack=AttackActionConfig(
            enabled=True,
            consumed_resources={"laser": 1},
            defense_resources={"armor": 1},
        ),
        swap=ActionConfig(enabled=True),
        # Disabled actions
        place_box=ActionConfig(enabled=False),
        change_color=ActionConfig(enabled=False),
        change_glyph=ChangeGlyphActionConfig(enabled=False, number_of_glyphs=4),
    )

    # Enhanced shaped rewards for transformer learning
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

    # Heart reward with maximum possible value
    env_cfg.game.agent.rewards.inventory.heart = 1
    env_cfg.game.agent.rewards.inventory.heart_max = 255

    # Easy converter configuration
    altar_copy = env_cfg.game.objects["altar"].model_copy(deep=True)
    altar_copy.input_resources = {"battery_red": 1}
    env_cfg.game.objects["altar"] = altar_copy

    # Set initial resource counts for immediate availability
    for obj_name in ["mine_red", "generator_red", "altar"]:
        if obj_name in env_cfg.game.objects:
            obj_copy = env_cfg.game.objects[obj_name].model_copy(deep=True)
            obj_copy.initial_resource_count = 1
            env_cfg.game.objects[obj_name] = obj_copy

    # Set label for clarity
    env_cfg.label = "arena.bes_transformer"

    # Configuration flags optimized for transformer learning
    env_cfg.desync_episodes = True
    env_cfg.game.track_movement_metrics = True
    env_cfg.game.no_agent_interference = False
    env_cfg.game.resource_loss_prob = 0.0
    env_cfg.game.recipe_details_obs = True  # Enable for richer observations

    # Enhanced global observation tokens for transformer
    env_cfg.game.global_obs.episode_completion_pct = True
    env_cfg.game.global_obs.last_action = True
    env_cfg.game.global_obs.last_reward = True
    env_cfg.game.global_obs.resource_rewards = True  # Enable for richer reward signals
    env_cfg.game.global_obs.visitation_counts = True  # Enable for spatial reasoning

    return env_cfg


def make_evals(env: Optional[EnvConfig] = None) -> List[SimulationConfig]:
    """Create evaluation environments for transformer testing."""
    basic_env = env or make_env()

    # Basic evaluation without combat
    basic_env.game.actions.attack.consumed_resources["laser"] = 100

    # Combat evaluation with attacks enabled
    combat_env = basic_env.model_copy()
    combat_env.game.actions.attack.consumed_resources["laser"] = 1

    # Extended evaluation for transformer capabilities
    memory_env = basic_env.model_copy()
    memory_env.label = "arena.bes_transformer.memory"
    memory_env.game.map_builder.width = 35  # Larger map for memory testing
    memory_env.game.map_builder.height = 35

    return [
        SimulationConfig(name="arena_bes_transformer/basic", env=basic_env),
        SimulationConfig(name="arena_bes_transformer/combat", env=combat_env),
        SimulationConfig(name="arena_bes_transformer/memory", env=memory_env),
    ]


def train() -> TrainTool:
    """
    Create training configuration optimized for transformer agent.

    This configuration is designed to leverage the transformer's
    enhanced capacity and GTrXL features with appropriate batch sizes
    and sequence lengths.
    """
    env_cfg = make_env()

    # Transformer-optimized trainer configuration
    trainer_cfg = TrainerConfig(
        curriculum=cc.env_curriculum(env_cfg),
        total_timesteps=5_000_000_000,  # Reduced for faster iteration
        # Transformer-optimized batch configuration
        batch_size=262144,  # Smaller batches for transformer memory efficiency
        minibatch_size=8192,  # Smaller minibatches
        bptt_horizon=128,  # Longer horizons for transformer context
        update_epochs=2,  # More epochs for complex learning
        checkpoint=CheckpointConfig(
            checkpoint_interval=25,  # More frequent checkpoints
            wandb_checkpoint_interval=25,
        ),
        evaluation=EvaluationConfig(
            simulations=make_evals(env_cfg),
            evaluate_remote=True,
            evaluate_local=False,
        ),
        # Enhanced performance settings for transformer
        compile=False,  # Keep disabled for stability during testing
        zero_copy=True,
        verbose=True,
    )

    return TrainTool(
        trainer=trainer_cfg,
        policy_architecture=AgentConfig(name="pytorch/transformer_improved"),
    )


def play(
    env: Optional[EnvConfig] = None,
    num_agents: int = 24,
    policy_uri: Optional[str] = None,
) -> PlayTool:
    """Interactive play tool for testing transformer agent."""
    if env is None:
        eval_env = make_env(num_agents=num_agents)
    else:
        eval_env = env

    return PlayTool(
        sim=SimulationConfig(env=eval_env, name="arena_bes_transformer"),
        policy_uri=policy_uri,
    )


def replay(env: Optional[EnvConfig] = None) -> ReplayTool:
    """Replay tool for viewing transformer agent episodes."""
    eval_env = env or make_env()
    return ReplayTool(sim=SimulationConfig(env=eval_env, name="arena_bes_transformer"))


def evaluate(policy_uri: str) -> SimTool:
    """Evaluate a trained transformer policy on arena environments."""
    return SimTool(
        simulations=make_evals(),
        policy_uris=[policy_uri],
    )


if __name__ == "__main__":
    """Allow running this recipe directly for play testing with transformer agent.
    
    Usage:
        uv run experiments/recipes/arena_bes_transformer.py [port] [num_agents_or_policy] [policy]
        
    Examples:
        uv run experiments/recipes/arena_bes_transformer.py              # Default: port 8001, 24 agents
        uv run experiments/recipes/arena_bes_transformer.py 8002 6        # Port 8002, 6 agents
        uv run experiments/recipes/arena_bes_transformer.py 8003 wandb://run/my-transformer-run
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
