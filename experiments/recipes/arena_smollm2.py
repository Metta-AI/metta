"""Arena training recipe with SmolLM2 agent."""

from typing import List, Optional

import metta.cogworks.curriculum as cc
import metta.mettagrid.builder.envs as eb
from metta.agent.agent_config import AgentConfig
from metta.cogworks.curriculum.curriculum import CurriculumConfig
from metta.mettagrid.mettagrid_config import MettaGridConfig
from metta.rl.loss.loss_config import LossConfig
from metta.rl.loss.ppo_config import PPOConfig
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


def make_tiny_mettagrid(num_agents: int = 4) -> MettaGridConfig:
    """Create a tiny arena environment for CPU debugging."""
    arena_env = eb.make_arena(num_agents=num_agents)
    # Very small map for CPU debugging
    arena_env.game.map_builder.width = 12
    arena_env.game.map_builder.height = 12
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

    # Create trainer configuration with minimal batch sizes for CPU debugging
    trainer_cfg = TrainerConfig(
        losses=LossConfig(
            loss_configs={
                "ppo": PPOConfig(
                    clip_coef=0.2,
                    ent_coef=0.01,
                    vf_coef=0.5,
                )
            }
        ),
        curriculum=curriculum or make_curriculum(),
        # Minimal batch sizes for CPU debugging
        batch_size=32,  # Minimal batch size for CPU
        minibatch_size=8,  # Minimal minibatch size
        rollout_workers=1,  # Single worker for CPU
        total_timesteps=10000,  # Very short run for debugging
        bptt_horizon=16,  # Reduced BPTT horizon for CPU
        forward_pass_minibatch_target_size=8,  # Minimal forward pass batch
        async_factor=1,  # No async for debugging
        evaluation=EvaluationConfig(
            simulations=[
                SimulationConfig(
                    name="arena_smollm2/basic",
                    env=make_mettagrid(),
                ),
            ],
            evaluate_interval=10000,  # Evaluate every 10k steps
        ),
    )

    return TrainTool(trainer=trainer_cfg, policy_architecture=agent_cfg)


def train_frozen() -> TrainTool:
    """Train with frozen LLM weights (only train heads)."""
    return train(freeze_llm=True)


def train_unfrozen() -> TrainTool:
    """Train with unfrozen LLM weights (full fine-tuning)."""
    return train(freeze_llm=False)


def train_cpu_debug() -> TrainTool:
    """Minimal configuration for CPU debugging on MacBook."""
    # Create tiny curriculum for CPU debugging
    tiny_env = make_tiny_mettagrid(num_agents=4)
    tiny_curriculum = make_curriculum(tiny_env)

    # Configure SmolLM2 agent with minimal settings
    agent_cfg = AgentConfig(
        name="pytorch/smollm2",
        clip_range=0.2,
        analyze_weights_interval=1000,  # Less frequent analysis
    )

    # Absolute minimal trainer configuration for CPU
    trainer_cfg = TrainerConfig(
        losses=LossConfig(
            loss_configs={
                "ppo": PPOConfig(
                    clip_coef=0.2,
                    ent_coef=0.01,
                    vf_coef=0.5,
                )
            }
        ),
        curriculum=tiny_curriculum,
        # Absolute minimal settings for CPU debugging
        batch_size=16,  # Smallest viable batch
        minibatch_size=4,  # Smallest minibatch
        rollout_workers=1,  # Single worker
        total_timesteps=5000,  # Very short debug run
        bptt_horizon=8,  # Minimal BPTT
        forward_pass_minibatch_target_size=4,  # Minimal forward batch
        async_factor=1,  # Synchronous operation
        update_epochs=1,  # Single epoch per update
        evaluation=EvaluationConfig(
            simulations=[
                SimulationConfig(
                    name="arena_smollm2/cpu_debug",
                    env=tiny_env,
                ),
            ],
            evaluate_interval=2500,  # Evaluate only once during run
            skip_git_check=True,  # Skip git check for debugging
        ),
    )

    return TrainTool(trainer=trainer_cfg, policy_architecture=agent_cfg)


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
