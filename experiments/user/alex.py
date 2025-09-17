# This file is for local experimentation only. It is not checked in, and therefore won't be usable on skypilot
# You can run these functions locally with e.g. `./tools/run.py experiments.recipes.scratchpad.alex.train`
# The VSCode "Run and Debug" section supports options to run these functions.
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
from metta.agent.policies.fast import FastConfig
from metta.rl.trainer_config import TrainerConfig
from metta.rl.training.evaluator import EvaluatorConfig
from metta.rl.training.training_environment import TrainingEnvironmentConfig
from metta.sim.simulation_config import SimulationConfig
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
from metta.tools.sim import SimTool
from metta.tools.train import TrainTool


def make_env(num_agents: int = 24) -> EnvConfig:
    """Create the pre-dehydration basic_easy_shaped environment."""

    env_cfg = eb.make_arena(num_agents=num_agents, combat=False)

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
                    "block": 20,
                    "wall": 20,
                },
            ),
        ),
    )

    env_cfg.game.objects["block"] = building.block

    env_cfg.game.actions = ActionsConfig(
        noop=ActionConfig(enabled=True),
        move_8way=ActionConfig(enabled=True),
        rotate=ActionConfig(enabled=True),
        put_items=ActionConfig(enabled=True),
        get_items=ActionConfig(enabled=True),
        attack=AttackActionConfig(
            enabled=True,
            consumed_resources={"laser": 1},
            defense_resources={"armor": 1},
        ),
        swap=ActionConfig(enabled=True),
        change_color=ActionConfig(enabled=False),
        change_glyph=ChangeGlyphActionConfig(enabled=False, number_of_glyphs=4),
    )

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
    env_cfg.game.agent.rewards.inventory.heart_max = 255

    altar_copy = env_cfg.game.objects["altar"].model_copy(deep=True)
    altar_copy.input_resources = {"battery_red": 1}
    env_cfg.game.objects["altar"] = altar_copy

    for obj_name in ["mine_red", "generator_red", "altar"]:
        if obj_name in env_cfg.game.objects:
            obj_copy = env_cfg.game.objects[obj_name].model_copy(deep=True)
            obj_copy.initial_resource_count = 1
            env_cfg.game.objects[obj_name] = obj_copy

    env_cfg.label = "arena.basic_easy_shaped"
    env_cfg.desync_episodes = True
    env_cfg.game.track_movement_metrics = True
    env_cfg.game.resource_loss_prob = 0.0
    env_cfg.game.recipe_details_obs = False

    env_cfg.game.global_obs.episode_completion_pct = True
    env_cfg.game.global_obs.last_action = True
    env_cfg.game.global_obs.last_reward = True
    env_cfg.game.global_obs.resource_rewards = False
    env_cfg.game.global_obs.visitation_counts = False

    return env_cfg


def make_evals(env: Optional[EnvConfig] = None) -> List[SimulationConfig]:
    basic_env = env or make_env()
    basic_env.game.actions.attack.consumed_resources["laser"] = 100

    combat_env = basic_env.model_copy()
    combat_env.game.actions.attack.consumed_resources["laser"] = 1

    return [
        SimulationConfig(name="arena_basic_easy_shaped/basic", env=basic_env),
        SimulationConfig(name="arena_basic_easy_shaped/combat", env=combat_env),
    ]


def train() -> TrainTool:
    env_cfg = make_env()

    trainer_cfg = TrainerConfig()
    trainer_cfg.total_timesteps = 10_000_000_000

    policy_config = FastConfig()
    policy_config.lstm_config.hidden_size = 128

    evaluator_cfg = EvaluatorConfig(
        simulations=make_evals(env_cfg),
        evaluate_remote=True,
        evaluate_local=False,
        skip_git_check=True,
    )
    evaluator_cfg.epoch_interval = 50

    return TrainTool(
        trainer=trainer_cfg,
        training_env=TrainingEnvironmentConfig(curriculum=cc.env_curriculum(env_cfg)),
        evaluator=evaluator_cfg,
        policy_architecture=policy_config,
    )


def play(
    env: Optional[EnvConfig] = None,
    num_agents: int = 24,
    policy_uri: Optional[str] = None,
) -> PlayTool:
    eval_env = env or make_env(num_agents=num_agents)
    return PlayTool(
        sim=SimulationConfig(env=eval_env, name="arena_basic_easy_shaped"),
        policy_uri=policy_uri,
    )


def replay(env: Optional[EnvConfig] = None) -> ReplayTool:
    eval_env = env or make_env()
    return ReplayTool(
        sim=SimulationConfig(env=eval_env, name="arena_basic_easy_shaped")
    )


def evaluate(policy_uri: str) -> SimTool:
    return SimTool(
        simulations=make_evals(),
        policy_uris=[policy_uri],
    )
