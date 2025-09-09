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
import json
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


class TribalNimPlayTool(Tool):
    """
    Hybrid tool that uses standard Simulation for neural network integration
    but routes visualization to native Nim instead of web-based mettascope.
    """
    
    env_config: TribalEnvConfig
    policy_uri: str | None = None

    def invoke(self, args: dict[str, str], overrides: list[str]) -> int | None:
        """Run tribal environment with neural network control via standard pipeline + Nim visualization."""
        print("🎮 Tribal Neural Network Play Tool")
        print("🧠 Using standard Simulation pipeline + Native Nim visualization")
        
        if self.policy_uri:
            return self._run_neural_network_with_nim_viewer()
        else:
            return self._run_nim_builtin_ai()

    def _run_neural_network_with_nim_viewer(self) -> int:
        """
        Use standard Simulation class for neural network integration,
        but extract actions and send to Nim visualization.
        """
        try:
            print("🔄 Setting up neural network pipeline...")
            
            # Import required modules
            import torch
            from metta.sim.simulation import Simulation
            from metta.sim.simulation_config import SimulationConfig
            from metta.sim.tribal_genny import TribalGridEnv
            import tribal
            
            # Create SimulationConfig from tribal environment config
            sim_config = SimulationConfig(
                name="tribal/neural_play",
                env=self.env_config,
                num_episodes=1,  # Single long episode for interactive play
                max_time_s=3600  # 1 hour max
            )
            
            # Create simulation using standard pipeline
            print("🏗️ Creating simulation with neural network...")
            simulation = Simulation.create(
                sim_config=sim_config,
                device="cpu",  # Use CPU for interactive play
                vectorization="serial",  # Single environment
                policy_uri=self.policy_uri,
                stats_dir="./train_dir/play_stats",
                replay_dir="./train_dir/play_replays"
            )
            
            print("✅ Neural network loaded via standard pipeline")
            
            # Initialize Nim external controller
            print("🔧 Initializing Nim external controller...")
            success = tribal.init_external_nncontroller()
            if not success:
                print("❌ Failed to initialize external controller")
                return 1
            
            print("✅ Nim external controller initialized")
            print(f"🤖 Controller type: {tribal.get_controller_type_string()}")
            
            # Start the action bridge loop
            print("🌉 Starting neural network → Nim action bridge...")
            return self._run_action_bridge(simulation, tribal)
            
        except Exception as e:
            print(f"❌ Error setting up neural network pipeline: {e}")
            return 1

    def _run_action_bridge(self, simulation, tribal_module) -> int:
        """
        Bridge actions from standard Simulation neural network to Nim environment.
        """
        try:
            print("🔄 Starting simulation...")
            simulation.start_simulation()
            simulation._policy.reset_memory()
            
            print("🎮 Action bridge active - neural network controlling Nim environment")
            print("💡 Launch Nim viewer now to see neural network in action!")
            
            step_count = 0
            max_steps = 1000  # Limit for interactive play
            
            while step_count < max_steps:
                # Generate actions using standard simulation pipeline
                actions_np = simulation.generate_actions()
                
                # Convert actions to format expected by Nim (SeqInt via genny bindings)
                # actions_np shape: [num_agents, 2] - [action_type, argument]
                actions_seq = tribal_module.SeqInt()
                for agent_idx in range(actions_np.shape[0]):
                    actions_seq.append(int(actions_np[agent_idx, 0]))  # action_type
                    actions_seq.append(int(actions_np[agent_idx, 1]))  # argument
                
                # Send actions to Nim environment
                success = tribal_module.set_external_actions_from_python(actions_seq)
                if not success:
                    print("⚠️ Failed to set actions in Nim environment")
                
                # Step the simulation (this updates observations for next iteration)
                simulation.step_simulation(actions_np)
                
                step_count += 1
                
                # Brief pause to allow Nim visualization to update
                import time
                time.sleep(0.1)  # 10 FPS
            
            print(f"✅ Completed {step_count} steps of neural network control")
            return 0
            
        except KeyboardInterrupt:
            print("\n⏹️ Stopping neural network control...")
            return 0
        except Exception as e:
            print(f"❌ Error in action bridge: {e}")
            return 1

    def _run_nim_builtin_ai(self) -> int:
        """
        Launch Nim environment with built-in AI (no neural network).
        """
        print("🎯 Launching Nim environment with built-in AI...")
        print("💡 No neural network specified - using built-in Nim AI")
        
        try:
            import tribal
            # Initialize built-in AI controller
            success = tribal.init_builtin_aicontroller()
            if success:
                print("✅ Built-in AI controller initialized")
                print(f"🤖 Controller type: {tribal.get_controller_type_string()}")
                print("🎮 Launch Nim viewer to see built-in AI in action!")
                return 0
            else:
                print("❌ Failed to initialize built-in AI controller")
                return 1
        except Exception as e:
            print(f"❌ Error initializing built-in AI: {e}")
            return 1


def tribal_env_curriculum(tribal_config: TribalEnvConfig) -> CurriculumConfig:
    """Create a curriculum configuration from a TribalEnvConfig - like cc.env_curriculum but for tribal"""
    task_gen_config = TribalTaskGeneratorConfig(env=tribal_config)
    return CurriculumConfig(task_generator=task_gen_config)


def make_tribal_environment(
    max_steps: int = None,
    enable_combat: bool = None,
    # Resource configuration
    ore_per_battery: int = None,
    batteries_per_heart: int = None,
    # Combat configuration
    clippy_spawn_rate: float = None,
    clippy_damage: int = None,
    # Reward configuration
    heart_reward: float = None,
    ore_reward: float = None,
    battery_reward: float = None,
    survival_penalty: float = None,
    death_penalty: float = None,
    **kwargs,
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
        ore_per_battery: Ore required to craft battery
        batteries_per_heart: Batteries required at altar for hearts
        clippy_spawn_rate: Rate of enemy spawning (0.0-1.0)
        clippy_damage: Damage dealt by enemies
        heart_reward: Reward for creating hearts
        ore_reward: Reward for collecting ore
        battery_reward: Reward for crafting batteries
        survival_penalty: Per-step survival penalty
        death_penalty: Penalty for agent death
        **kwargs: Additional configuration overrides for TribalEnvConfig (not game config)
    """
    from metta.sim.tribal_genny import TribalGameConfig

    # Build game configuration with overrides
    game_overrides = {}
    if max_steps is not None:
        game_overrides["max_steps"] = max_steps
    if enable_combat is not None:
        game_overrides["enable_combat"] = enable_combat
    if ore_per_battery is not None:
        game_overrides["ore_per_battery"] = ore_per_battery
    if batteries_per_heart is not None:
        game_overrides["batteries_per_heart"] = batteries_per_heart
    if clippy_spawn_rate is not None:
        game_overrides["clippy_spawn_rate"] = clippy_spawn_rate
    if clippy_damage is not None:
        game_overrides["clippy_damage"] = clippy_damage
    if heart_reward is not None:
        game_overrides["heart_reward"] = heart_reward
    if ore_reward is not None:
        game_overrides["ore_reward"] = ore_reward
    if battery_reward is not None:
        game_overrides["battery_reward"] = battery_reward
    if survival_penalty is not None:
        game_overrides["survival_penalty"] = survival_penalty
    if death_penalty is not None:
        game_overrides["death_penalty"] = death_penalty

    # Create game config with overrides
    game_config = TribalGameConfig(**game_overrides)

    # Create environment config with custom game config
    config = TribalEnvConfig(
        label="tribal_basic", desync_episodes=True, game=game_config, **kwargs
    )

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
) -> "TribalNimPlayTool":
    """
    Interactive play with the tribal environment using native Nim visualization.

    This avoids the web-based mettascope viewer and uses Nim's native rendering.
    When policy_uri is provided, the Nim environment will accept external actions
    from the trained neural network. Otherwise it falls back to built-in AI.

    Args:
        env: Optional tribal environment config
        policy_uri: Optional URI to trained policy
        **overrides: Additional configuration overrides
    """
    # Use the configured tribal environment, or create default
    play_env = env or make_tribal_environment()

    return TribalNimPlayTool(
        env_config=play_env,
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
