"""
Tribal Basic Recipe - Nim Environment Integration

SETUP REQUIREMENTS:
Bindings are built automatically when the recipe runs!

TECHNICAL NOTES:
- Uses genny-generated Nim bindings for high performance (10-100x faster than JSON IPC)
- Agent count (15), map size (100x50), observation shape (19 layers, 11x11) are compile-time constants
- GPU compatibility: Works with RTX 5080 using PufferLib rebuilt with TORCH_CUDA_ARCH_LIST="8.0"
- Bindings are built automatically and configuration uses updated clippy spawn rate (1.0)
"""

import subprocess
import sys
import os
from pathlib import Path

from metta.cogworks.curriculum.curriculum import CurriculumConfig
from metta.common.tool import Tool
from metta.cogworks.curriculum.task_generator import TaskGeneratorConfig
from metta.rl.trainer_config import TrainerConfig, EvaluationConfig
from metta.rl.loss.loss_config import LossConfig
from metta.sim.simulation_config import SimulationConfig
from metta.sim.tribal_genny import TribalEnvConfig
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
from metta.tools.sim import SimTool
from metta.tools.train import TrainTool


def _ensure_tribal_bindings_built():
    """
    Automatically build tribal bindings if they don't exist or are outdated.
    """
    # Find the metta project root and tribal directory
    metta_root = Path(__file__).parent.parent.parent
    tribal_dir = metta_root / "tribal"
    bindings_dir = tribal_dir / "bindings" / "generated"

    # Check if bindings exist
    library_files = list(bindings_dir.glob("libtribal.*"))
    python_binding = bindings_dir / "tribal.py"

    needs_build = (
        not bindings_dir.exists()
        or not python_binding.exists()
        or len(library_files) == 0
    )

    if needs_build:
        print("🔧 Building tribal bindings...")
        build_script = tribal_dir / "build_bindings.sh"

        if not build_script.exists():
            raise FileNotFoundError(f"Tribal build script not found at {build_script}")

        # Run the build script
        result = subprocess.run(
            ["bash", str(build_script)],
            cwd=str(tribal_dir),
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print("❌ Failed to build tribal bindings:")
            print(f"stdout: {result.stdout}")
            print(f"stderr: {result.stderr}")
            raise RuntimeError("Tribal bindings build failed")

        print("✅ Tribal bindings built successfully")

    # Add bindings to Python path if not already there
    bindings_path = str(bindings_dir)
    if bindings_path not in sys.path:
        sys.path.insert(0, bindings_path)


class TribalTaskGeneratorConfig(TaskGeneratorConfig):
    """Simple tribal-specific task generator config - recipe-local only"""

    env: TribalEnvConfig

    def create(self):
        """Create a simple task generator that always returns the same tribal env"""
        from metta.cogworks.curriculum.task_generator import TaskGenerator
        import random

        class SimpleTribalTaskGenerator(TaskGenerator):
            def __init__(self, config):
                super().__init__(config)
                self._env = config.env

            def _generate_task(self, task_id: int, rng: random.Random):
                return self._env.model_copy(deep=True)

        return SimpleTribalTaskGenerator(self)


class NimPlayTool(Tool):
    """Tool for launching the Nim-based tribal viewer directly."""
    
    env_config: TribalEnvConfig
    policy_uri: str | None = None

    def invoke(self, args: dict[str, str], overrides: list[str]) -> int | None:
        """Launch the Nim tribal viewer."""
        # Find the metta project root and tribal directory
        metta_root = Path(__file__).parent.parent.parent
        tribal_dir = metta_root / "tribal"
        
        # Ensure we're in the right directory
        if not tribal_dir.exists():
            print(f"❌ Tribal directory not found at {tribal_dir}")
            return 1
        
        print("🎮 Launching Nim tribal viewer...")
        print(f"📁 Working directory: {tribal_dir}")
        
        # Build and run the Nim visualization
        try:
            # Change to tribal directory
            original_dir = os.getcwd()
            os.chdir(tribal_dir)
            
            # Run the Nim viewer using nimble
            result = subprocess.run(
                ["nimble", "visualize"],
                capture_output=False,  # Let output show directly
                text=True
            )
            
            return result.returncode
        
        except KeyboardInterrupt:
            print("\n🛑 Nim viewer stopped by user")
            return 0
        except Exception as e:
            print(f"❌ Error launching Nim viewer: {e}")
            return 1
        finally:
            # Restore original directory
            try:
                os.chdir(original_dir)
            except:
                pass


def tribal_env_curriculum(tribal_config: TribalEnvConfig) -> CurriculumConfig:
    """Create a curriculum configuration from a TribalEnvConfig - like cc.env_curriculum but for tribal"""
    task_gen_config = TribalTaskGeneratorConfig(env=tribal_config)
    return CurriculumConfig(task_generator=task_gen_config)


def make_tribal_environment(
    max_steps: int = None, enable_combat: bool = None, **kwargs
) -> TribalEnvConfig:
    """
    Create tribal environment configuration for training.

    The tribal environment features:
    - Village-based agent tribes with shared altars (15 agents, compile-time constant)
    - Multi-step resource chains (ore → battery → hearts)
    - Crafting system (wood → spears, wheat → hats/food, ore → armor)
    - Defensive gameplay against Clippy enemies
    - Terrain interaction (water, wheat fields, forests)

    NOTE: Agent count, map dimensions, and observation space are compile-time constants
    for performance. Only gameplay parameters are configurable.

    Args:
        max_steps: Maximum steps per episode (uses Nim default if None)
        enable_combat: Enable combat with Clippys (uses Nim default if None)
        **kwargs: Additional configuration overrides
    """
    # Create with default factory
    config = TribalEnvConfig(label="tribal_basic", desync_episodes=True, **kwargs)

    # Override only specified parameters
    if max_steps is not None:
        config.game.max_steps = max_steps
    if enable_combat is not None:
        config.game.enable_combat = enable_combat

    # Note: heart_reward, battery_reward, ore_reward now come from Nim defaults
    # No longer hardcoded here - single source of truth in Nim

    return config


def train() -> TrainTool:
    """
    Train agents on the tribal environment.

    Uses a minimal configuration similar to the working arena recipe.
    Automatically builds tribal bindings if needed.
    """
    # Ensure tribal bindings are built
    _ensure_tribal_bindings_built()

    # Create environment (uses compile-time constant: 15 agents)
    env = make_tribal_environment()

    # Create curriculum with tribal environment (like cc.env_curriculum but for tribal)
    curriculum = tribal_env_curriculum(env)

    # Minimal trainer config like arena recipe
    trainer_config = TrainerConfig(
        losses=LossConfig(),
        curriculum=curriculum,
        evaluation=EvaluationConfig(
            simulations=[
                SimulationConfig(name="tribal/basic", env=env),
            ],
            skip_git_check=True,  # Skip git check for development
        ),
    )

    return TrainTool(trainer=trainer_config)


def evaluate(
    policy_uri: str, run: str = "tribal_eval", num_episodes: int = 10, **overrides
) -> SimTool:
    """
    Evaluate a trained policy on the tribal environment.

    Args:
        policy_uri: URI to trained policy (file:// or wandb://)
        run: Name for this evaluation run
        num_episodes: Number of episodes to evaluate
        **overrides: Additional configuration overrides
    """
    # Ensure tribal bindings are built
    _ensure_tribal_bindings_built()

    env = make_tribal_environment()

    return SimTool(
        env=env, policy_uri=policy_uri, num_episodes=num_episodes, run=run, **overrides
    )


def play(
    env: TribalEnvConfig | None = None, policy_uri: str | None = None, **overrides
) -> "NimPlayTool":
    """
    Interactive play with the tribal environment using direct Nim execution.

    This version launches the Nim-based tribal viewer directly, bypassing the Python
    play infrastructure for better performance and native Nim rendering.

    Args:
        env: Optional tribal environment config (unused in Nim version)
        policy_uri: Optional URI to trained policy (future feature)
        **overrides: Additional configuration overrides
    """
    return NimPlayTool(
        env_config=env or make_tribal_environment(),
        policy_uri=policy_uri,
        **overrides,
    )


def replay(policy_uri: str, **overrides) -> ReplayTool:
    """
    Replay recorded tribal episodes.

    Args:
        policy_uri: URI to policy that generated replays
        **overrides: Additional configuration overrides
    """
    # Ensure tribal bindings are built
    _ensure_tribal_bindings_built()

    env = make_tribal_environment()

    return ReplayTool(
        sim=SimulationConfig(name="tribal/replay", env=env),
        policy_uri=policy_uri,
        **overrides,
    )
