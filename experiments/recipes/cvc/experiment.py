"""Programmatic helpers and Skypilot launch utilities for CVC recipes."""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from typing import Sequence

from experiments.recipes.cvc.coordination import train as train_coordination
from experiments.recipes.cvc.curriculum import (
    make_curriculum,
    make_training_env,
    train as train_curriculum,
)
from experiments.recipes.cvc.evaluation import evaluate, make_eval_suite
from experiments.recipes.cvc.medium_maps import train as train_medium_maps
from experiments.recipes.cvc.core import play
from experiments.recipes.cvc.single_mission import train as train_single_mission
from experiments.recipes.cvc.small_maps import train as train_small_maps
from metta.sim.simulation_config import SimulationConfig
from metta.tools.eval import EvaluateTool
from metta.tools.play import PlayTool
from metta.tools.train import TrainTool
from mettagrid.config.mettagrid_config import MettaGridConfig

RECIPE_MODULE = "experiments.recipes.cvc"

__all__ = [
    "SkypilotExperiment",
    "EXPERIMENTS",
    "DEFAULT_EXPERIMENTS",
    "basic_training",
    "medium_training",
    "custom_curriculum",
    "single_mission_debug",
    "evaluation",
    "multi_agent_coordination",
    "custom_eval_suite",
    "play_trained_policy",
    "inspect_training_env",
    "experiment",
]


@dataclass(frozen=True)
class SkypilotExperiment:
    """Configuration for launching a Skypilot job."""

    tool_suffix: str
    num_cogs: int
    gpus: int
    timesteps: int
    mission_name: str | None = None

    @property
    def tool_path(self) -> str:
        return f"{RECIPE_MODULE}.{self.tool_suffix}"

    def build_command(
        self,
        *,
        run_name: str,
        heartbeat_timeout: int,
        skip_git_check: bool,
    ) -> list[str]:
        """Create the command line for this experiment."""
        cmd = [
            "./devops/skypilot/launch.py",
            self.tool_path,
            f"run={run_name}",
            f"num_cogs={self.num_cogs}",
            f"trainer.total_timesteps={self.timesteps}",
            f"--gpus={self.gpus}",
            f"--heartbeat-timeout={heartbeat_timeout}",
        ]
        if self.mission_name:
            cmd.insert(3, f"mission_name={self.mission_name}")
        if skip_git_check:
            cmd.append("--skip-git-check")
        return cmd


EXPERIMENTS: dict[str, SkypilotExperiment] = {
    "debug_single": SkypilotExperiment(
        tool_suffix="single_mission.train",
        mission_name="extractor_hub_30",
        num_cogs=2,
        gpus=1,
        timesteps=5_000_000,
    ),
    "small_1cog": SkypilotExperiment(
        tool_suffix="small_maps.train",
        num_cogs=1,
        gpus=2,
        timesteps=20_000_000,
    ),
    "small_2cogs": SkypilotExperiment(
        tool_suffix="small_maps.train",
        num_cogs=2,
        gpus=2,
        timesteps=20_000_000,
    ),
    "small_4cogs": SkypilotExperiment(
        tool_suffix="small_maps.train",
        num_cogs=4,
        gpus=4,
        timesteps=30_000_000,
    ),
    "medium_4cogs": SkypilotExperiment(
        tool_suffix="medium_maps.train",
        num_cogs=4,
        gpus=4,
        timesteps=40_000_000,
    ),
    "coordination_4cogs": SkypilotExperiment(
        tool_suffix="coordination.train",
        num_cogs=4,
        gpus=4,
        timesteps=40_000_000,
    ),
    "full_1cog": SkypilotExperiment(
        tool_suffix="curriculum.train",
        num_cogs=1,
        gpus=4,
        timesteps=50_000_000,
    ),
    "full_4cogs": SkypilotExperiment(
        tool_suffix="curriculum.train",
        num_cogs=4,
        gpus=8,
        timesteps=100_000_000,
    ),
    "full_8cogs": SkypilotExperiment(
        tool_suffix="curriculum.train",
        num_cogs=8,
        gpus=8,
        timesteps=100_000_000,
    ),
}

DEFAULT_EXPERIMENTS: tuple[str, ...] = tuple(
    name for name in EXPERIMENTS if not name.startswith("debug")
)


def basic_training(*, num_cogs: int = 4) -> TrainTool:
    """Train on small maps with a configurable number of agents."""
    return train_small_maps(num_cogs=num_cogs)


def medium_training(*, num_cogs: int = 4) -> TrainTool:
    """Train on medium maps with a configurable number of agents."""
    return train_medium_maps(num_cogs=num_cogs)


def custom_curriculum() -> TrainTool:
    """Create a custom curriculum tool with additional logging."""
    curriculum = make_curriculum(
        num_cogs=4,
        base_missions=["extractor_hub_30", "oxygen_bottleneck", "energy_starved"],
        enable_detailed_slice_logging=True,
    )
    return train_curriculum(num_cogs=4, curriculum=curriculum)


def single_mission_debug() -> TrainTool:
    """Create a debugging tool for a single mission."""
    return train_single_mission(mission_name="extractor_hub_30", num_cogs=2)


def evaluation() -> EvaluateTool:
    """Create an evaluation tool for the default checkpoints."""
    return evaluate(
        policy_uris=["file://./checkpoints/cvc_default/latest"],
        num_cogs=4,
        difficulty="standard",
    )


def multi_agent_coordination(*, num_cogs: int = 4) -> TrainTool:
    """Train specifically on multi-agent coordination."""
    return train_coordination(num_cogs=num_cogs)


def custom_eval_suite() -> Sequence[SimulationConfig]:
    """Create a custom evaluation suite."""
    suite = make_eval_suite(
        num_cogs=8,
        difficulty="standard",
        subset=["extractor_hub_30", "extractor_hub_50", "extractor_hub_70"],
    )
    print(f"Created suite with {len(suite)} simulations:")
    for sim in suite:
        print(f"  - {sim.name}")
    return suite


def play_trained_policy() -> PlayTool:
    """Play a trained policy interactively."""
    return play(
        policy_uri="file://./checkpoints/cvc_default/latest",
        mission_name="extractor_hub_30",
        num_cogs=4,
    )


def inspect_training_env() -> MettaGridConfig:
    """Inspect the configuration of a training environment."""
    env = make_training_env(num_cogs=4, mission_name="extractor_hub_30")

    print("Environment Configuration:")
    print(f"  Num agents: {env.game.num_agents}")
    print(f"  Max steps: {env.game.max_steps}")

    print("\nStation Efficiencies:")
    for obj_name in [
        "charger",
        "carbon_extractor",
        "oxygen_extractor",
        "silicon_extractor",
    ]:
        if obj_name in env.game.objects:
            obj = env.game.objects[obj_name]
            if hasattr(obj, "efficiency"):
                print(f"  {obj_name}: {obj.efficiency}%")

    print("\nReward Structure:")
    print(f"  Inventory rewards: {env.game.agent.rewards.inventory}")
    return env


def experiment(
    configs: Sequence[str] | None = None,
    heartbeat_timeout: int = 3600,
    skip_git_check: bool = True,
) -> None:
    """Launch Skypilot jobs for multiple training configurations."""
    target_configs = list(configs or DEFAULT_EXPERIMENTS)

    print(f"Launching {len(target_configs)} Skypilot job(s):")
    for name in target_configs:
        print(f"  - {name}")

    for name in target_configs:
        config = EXPERIMENTS.get(name)
        if config is None:
            print(f"Warning: Unknown config '{name}', skipping")
            continue

        run_name = f"cvc_{name}.{time.strftime('%Y-%m-%d_%H%M')}"
        cmd_args = config.build_command(
            run_name=run_name,
            heartbeat_timeout=heartbeat_timeout,
            skip_git_check=skip_git_check,
        )

        print(f"\nLaunching: {name}")
        print(f"  Run: {run_name}")
        print(f"  Tool: {config.tool_path}")
        print(
            f"  Agents: {config.num_cogs}, GPUs: {config.gpus}, Steps: {config.timesteps:,}"
        )

        subprocess.run(cmd_args, check=False)
        time.sleep(1)

    print(f"\n✓ Successfully launched {len(target_configs)} jobs")


if __name__ == "__main__":
    import sys

    # Allow passing config names as command line arguments
    # e.g., python experiment.py debug_single small_4cogs
    if len(sys.argv) > 1:
        configs = sys.argv[1:]
        print(f"Running experiments: {', '.join(configs)}")
        experiment(configs=configs)
    else:
        # Default: run all non-debug experiments
        print("Running all standard experiments (not debug)")
        experiment()
