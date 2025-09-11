from typing import List, Optional

import metta.cogworks.curriculum as cc
import metta.map.scenes.random
import metta.mettagrid.builder.envs as eb
from metta.map.mapgen import MapGen
from metta.mettagrid.builder import building
from metta.mettagrid.mettagrid_config import (
    ActionConfig,
    ActionsConfig,
    AttackActionConfig,
    ChangeGlyphActionConfig,
    EnvConfig,
)
from metta.rl.loss.loss_config import LossConfig
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
    """Create the pre-dehydration basic_easy_shaped environment."""

    # Start with standard arena configuration
    env_cfg = eb.make_arena(num_agents=num_agents, combat=False)

    # Map configuration from original configs/env/mettagrid/arena/basic.yaml
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
                    "block": 20,  # Blocks included for environment variety
                    "wall": 20,  # 20 walls for proper spacing
                },
            ),
        ),
    )

    # Add block object from original configuration
    env_cfg.game.objects["block"] = building.block

    # Keep combat buildings enabled (lasery and armory stay in the config)

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

    # Heart reward with maximum possible value
    env_cfg.game.agent.rewards.inventory.heart = 1
    env_cfg.game.agent.rewards.inventory.heart_max = 255

    # Easy converter configuration (from configs/env/mettagrid/game/objects/basic_easy.yaml)
    # Altar only needs 1 battery_red instead of 3
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
    env_cfg.label = "arena.basic_easy_shaped"

    # Global configuration flags from old mettagrid.yaml
    env_cfg.desync_episodes = True  # Changes max_steps for first episode only
    env_cfg.game.track_movement_metrics = True
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

    # Create trainer configuration, only overriding non-default values
    trainer_cfg = TrainerConfig(
        losses=LossConfig(),
        curriculum=cc.env_curriculum(env_cfg),
        total_timesteps=10_000_000_000,  # 10B instead of default 50B
        checkpoint=CheckpointConfig(
            checkpoint_interval=50,  # 50 instead of default 5
            wandb_checkpoint_interval=50,  # 50 instead of default 5
        ),
        evaluation=EvaluationConfig(
            simulations=make_evals(env_cfg),
            evaluate_remote=True,  # True instead of default False
            evaluate_local=False,  # False instead of default True
            skip_git_check=True,
        ),
        # All optimizer, PPO, and batch settings use defaults which already match
        # the original trainer.yaml configuration
    )

    return TrainTool(trainer=trainer_cfg)


def play(
    env: Optional[EnvConfig] = None,
    num_agents: int = 24,
    policy_uri: Optional[str] = None,
) -> PlayTool:
    """Interactive play tool for testing the environment."""
    if env is None:
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


# def play() -> PlayTool:
#     env = arena.make_evals()[0].env
#     env.game.max_steps = 100
#     cfg = arena.play(env)
#     return cfg


# def replay() -> ReplayTool:
#     env = arena.make_env()
#     env.game.max_steps = 100
#     cfg = arena.replay(env)
#     # cfg.policy_uri = "wandb://run/daveey.combat.lpsm.8x4"
#     return cfg


# def evaluate(run: str = "local.alex.1") -> SimTool:
#     cfg = arena.evaluate(policy_uri=f"wandb://run/{run}")

#     # If your run doesn't exist, try this:
#     # cfg = arena.evaluate(policy_uri="wandb://run/daveey.combat.lpsm.8x4")
#     return cfg
