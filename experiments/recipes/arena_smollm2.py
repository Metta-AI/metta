"""Arena training recipe with SmolLM2 agent."""

from typing import List, Optional

import metta.cogworks.curriculum as cc
import metta.mettagrid.builder.envs as eb
from metta.agent.agent_config import AgentConfig
from metta.cogworks.curriculum.curriculum import CurriculumConfig
from metta.mettagrid.mettagrid_config import MettaGridConfig
from metta.rl.loss.loss_config import LossConfig
from metta.rl.trainer_config import EvaluationConfig, TrainerConfig
from metta.sim.simulation_config import SimulationConfig
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
from metta.tools.sim import SimTool
from metta.tools.train import TrainTool


def make_mettagrid(num_agents: int = 12) -> MettaGridConfig:
    """Create a smaller arena environment for LLM testing."""
    arena_env = eb.make_arena(num_agents=num_agents)
    # Smaller map for faster training
    arena_env.game.map_builder.width = 20
    arena_env.game.map_builder.height = 20
    return arena_env


def make_curriculum(arena_env: Optional[MettaGridConfig] = None) -> CurriculumConfig:
    """Create a simplified curriculum for initial testing."""
    arena_env = arena_env or make_mettagrid()

    # Create simplified training tasks
    arena_tasks = cc.bucketed(arena_env)

    # Add some basic reward variations
    for item in ["ore_red", "battery_red"]:
        arena_tasks.add_bucket(f"game.agent.rewards.inventory.{item}", [0, 0.5, 1.0])

    return CurriculumConfig(task_generator=arena_tasks)


def make_evals(env: Optional[MettaGridConfig] = None) -> List[SimulationConfig]:
    """Create evaluation environments."""
    basic_env = env or make_mettagrid()

    return [
        SimulationConfig(name="arena_smollm2/basic", env=basic_env),
    ]


def train(
    curriculum: Optional[CurriculumConfig] = None,
    freeze_llm: bool = True,
) -> TrainTool:
    """Train SmolLM2 agent on arena environment.

    Args:
        curriculum: Optional curriculum config
        freeze_llm: Whether to freeze the LLM weights (default: True for faster training)
    """
    # Configure SmolLM2 agent
    agent_cfg = AgentConfig(
        name="pytorch/smollm2",
        clip_range=0.2,  # Moderate weight clipping for stability
        analyze_weights_interval=500,
    )

    # Create trainer configuration with smaller batch sizes for LLM
    trainer_cfg = TrainerConfig(
        agent=agent_cfg,
        losses=LossConfig(
            ppo_clip=0.2,  # Standard PPO clipping
            entropy=0.01,  # Moderate entropy for exploration
            value=0.5,  # Value loss coefficient
        ),
        curriculum=curriculum or make_curriculum(),
        # Smaller batch sizes for LLM training
        batch_size=256,  # Reduced from default
        minibatch_size=32,  # Smaller minibatches
        num_workers=4,  # Fewer workers for initial testing
        total_timesteps=100000,  # Short initial training run
        learning_rate=3e-4,  # Standard LR for fine-tuning
        evaluation=EvaluationConfig(
            simulations=[
                SimulationConfig(
                    name="arena_smollm2/basic",
                    env=make_mettagrid(),
                ),
            ],
            interval=10000,  # Evaluate every 10k steps
        ),
    )

    return TrainTool(trainer=trainer_cfg)


def train_frozen() -> TrainTool:
    """Train with frozen LLM weights (only train heads)."""
    return train(freeze_llm=True)


def train_unfrozen() -> TrainTool:
    """Train with unfrozen LLM weights (full fine-tuning)."""
    return train(freeze_llm=False)


def evaluate(
    policy_uri: str,
    simulations: Optional[List[SimulationConfig]] = None,
) -> SimTool:
    """Evaluate a trained SmolLM2 policy."""
    return SimTool(
        policy_uri=policy_uri,
        simulations=simulations or make_evals(),
    )


def play(policy_uri: str, env: Optional[MettaGridConfig] = None) -> PlayTool:
    """Interactive play with SmolLM2 agent."""
    return PlayTool(
        policy_uri=policy_uri,
        env=env or make_mettagrid(),
    )


def replay(policy_uri: str) -> ReplayTool:
    """Watch replays of SmolLM2 agent."""
    return ReplayTool(policy_uri=policy_uri)


def analyze(eval_db_uri: str):
    """Analyze evaluation results."""
    from metta.tools.analyze import AnalyzeTool

    return AnalyzeTool(eval_db_uri=eval_db_uri)
