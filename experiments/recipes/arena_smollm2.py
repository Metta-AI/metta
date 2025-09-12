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
    """Create a smaller arena environment with easy shaped rewards for LLM training."""
    arena_env = eb.make_arena(num_agents=num_agents)
    # Smaller map for faster training
    arena_env.game.map_builder.width = 20
    arena_env.game.map_builder.height = 20

    # Apply easy shaped rewards from arena_basic_easy_shaped
    arena_env.game.agent.rewards.inventory = {
        "heart": 1,
        "ore_red": 0.1,
        "battery_red": 0.8,
        "laser": 0.5,
        "armor": 0.5,
        "blueprint": 0.5,
    }
    arena_env.game.agent.rewards.inventory_max = {
        "heart": 100,
        "ore_red": 1,
        "battery_red": 1,
        "laser": 1,
        "armor": 1,
        "blueprint": 1,
    }

    # Easy converter: 1 battery_red to 1 heart (simplified from 3 to 1)
    arena_env.game.objects["altar"].input_resources = {"battery_red": 1}

    return arena_env


def make_tiny_mettagrid(num_agents: int = 6) -> MettaGridConfig:
    """Create a tiny arena environment for CPU debugging with easy shaped rewards."""
    arena_env = eb.make_arena(num_agents=num_agents)
    # Very small map for CPU debugging
    arena_env.game.map_builder.width = 12
    arena_env.game.map_builder.height = 12

    # Apply same easy shaped rewards as regular environment
    arena_env.game.agent.rewards.inventory = {
        "heart": 1,
        "ore_red": 0.1,
        "battery_red": 0.8,
        "laser": 0.5,
        "armor": 0.5,
        "blueprint": 0.5,
    }
    arena_env.game.agent.rewards.inventory_max = {
        "heart": 100,
        "ore_red": 1,
        "battery_red": 1,
        "laser": 1,
        "armor": 1,
        "blueprint": 1,
    }

    # Easy converter: 1 battery_red to 1 heart
    arena_env.game.objects["altar"].input_resources = {"battery_red": 1}

    return arena_env


def make_curriculum(arena_env: Optional[MettaGridConfig] = None) -> CurriculumConfig:
    """Create curriculum with easy shaped reward variations for LLM training."""
    arena_env = arena_env or make_mettagrid()

    # Create training tasks with curriculum from arena_basic_easy_shaped
    arena_tasks = cc.bucketed(arena_env)

    # Add reward variations for key items (similar to arena_basic_easy_shaped)
    for item in ["ore_red", "battery_red", "laser", "armor"]:
        arena_tasks.add_bucket(
            f"game.agent.rewards.inventory.{item}", [0, 0.1, 0.5, 0.9, 1.0]
        )
        arena_tasks.add_bucket(f"game.agent.rewards.inventory.{item}_max", [1, 2])

    # Enable/disable attacks using cost to maintain action space consistency
    arena_tasks.add_bucket("game.actions.attack.consumed_resources.laser", [1, 100])

    # Sometimes add initial items to buildings for easier bootstrapping
    for obj in ["mine_red", "generator_red", "altar", "lasery", "armory"]:
        arena_tasks.add_bucket(f"game.objects.{obj}.initial_resource_count", [0, 1])

    return CurriculumConfig(task_generator=arena_tasks)


def make_evals(env: Optional[MettaGridConfig] = None) -> List[SimulationConfig]:
    """Create evaluation environments (basic and combat variants)."""
    basic_env = env or make_mettagrid()

    # Basic environment with disabled combat
    basic_env.game.actions.attack.consumed_resources["laser"] = 100

    # Combat environment with enabled combat
    combat_env = basic_env.model_copy()
    combat_env.game.actions.attack.consumed_resources["laser"] = 1

    return [
        SimulationConfig(name="arena_smollm2/basic", env=basic_env),
        SimulationConfig(name="arena_smollm2/combat", env=combat_env),
    ]


def train(
    curriculum: Optional[CurriculumConfig] = None,
    freeze_llm: bool = True,
) -> TrainTool:
    """Train SmolLM2 agent on arena environment.

    Uses smaller batch sizes optimized for SmolLM2 stability and memory efficiency.
    Reduced batch sizes help avoid tensor processing issues specific to LLM agents.
    For CPU debugging with minimal resources, use train_cpu_debug().

    Args:
        curriculum: Optional curriculum config
        freeze_llm: Whether to freeze the LLM weights (default: True for faster training)
    """
    # Configure SmolLM2 agent
    agent_cfg = AgentConfig(
        name="pytorch/smollm2",
        clip_range=0.2,  # Moderate weight clipping for stability
        analyze_weights_interval=2000,  # Less frequent analysis to save memory
    )

    # Ultra memory-efficient configuration for RTX 4090 constraints
    # Supports up to 8192 agents (131072 / 16 bptt_horizon) with aggressive memory optimization
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
        # Very small batch sizes for SmolLM2 memory constraints
        # Aggressive reduction to avoid CUDA OOM with 135M parameter LLM
        batch_size=4096,  # 4K batch - much more conservative for LLM memory usage
        minibatch_size=128,  # Very small minibatches to fit in GPU memory
        bptt_horizon=4,  # Minimal context for memory efficiency
        forward_pass_minibatch_target_size=16,  # Tiny forward passes
        update_epochs=8,  # More epochs to compensate for smaller batches
        async_factor=1,  # Disable async for memory savings
        compile=False,  # Disable torch.compile to save memory
        evaluation=EvaluationConfig(
            simulations=make_evals(),
            evaluate_interval=100,  # Less frequent evaluation to save memory
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
    tiny_env = make_tiny_mettagrid(num_agents=6)
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
        batch_size=24,  # Even smaller - 6 agents * 4 BPTT horizon
        minibatch_size=4,  # Much smaller minibatches
        rollout_workers=1,  # Single worker
        total_timesteps=5000,  # Very short debug run
        bptt_horizon=4,  # Even shorter BPTT
        forward_pass_minibatch_target_size=2,  # Minimal forward batch
        async_factor=1,  # Synchronous operation
        update_epochs=1,  # Single epoch per update
        evaluation=EvaluationConfig(
            simulations=make_evals(tiny_env),
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
