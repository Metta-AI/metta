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
import time
from pathlib import Path

# Add project root to Python path for tribal module imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from metta.cogworks.curriculum.curriculum import CurriculumConfig
from metta.common.tool import Tool
from metta.cogworks.curriculum.task_generator import TaskGeneratorConfig
from metta.rl.trainer_config import TrainerConfig, EvaluationConfig
from metta.rl.loss.loss_config import LossConfig
from metta.sim.simulation_config import SimulationConfig
from tribal.src.tribal_genny import TribalEnvConfig
from metta.tools.replay import ReplayTool
from metta.tools.sim import SimTool
from metta.tools.train import TrainTool


class DirectGennyTribalEnv:
    """
    Wrapper that allows direct genny binding calls with simple Python data structures.
    
    Instead of numpy arrays, this accepts simple Python lists and calls genny bindings directly.
    This avoids the numpy → SeqInt conversion layer for better performance.
    """
    
    def __init__(self, tribal_grid_env):
        self.tribal_env = tribal_grid_env
        self.num_agents = tribal_grid_env.num_agents
        self.max_steps = tribal_grid_env.max_steps
        
        # Get direct access to the underlying Nim environment via genny bindings
        self._nim_env = tribal_grid_env._nim_env
        
        # Initialize ExternalNN controller for Python neural network control
        # Import the controller functions from the tribal bindings
        import sys
        from pathlib import Path
        bindings_path = str(Path(__file__).parent.parent / "tribal" / "bindings" / "generated")
        if bindings_path not in sys.path:
            sys.path.insert(0, bindings_path)
        
        import tribal
        print("🔧 Initializing ExternalNN controller for Python neural network control...")
        success = tribal.init_external_nncontroller()
        if success:
            print("✅ ExternalNN controller initialized successfully")
        else:
            print("❌ Failed to initialize ExternalNN controller")
            raise RuntimeError("Failed to initialize ExternalNN controller")
    
    def reset(self, seed=None):
        """Reset and return observations."""
        return self.tribal_env.reset(seed)
    
    def step(self, actions):
        """
        Step with actions - accepts either:
        - Python list of lists: [[0,0], [1,2], [0,0], ...] (preferred for genny)
        - Numpy array: np.array([[0,0], [1,2], [0,0], ...])
        """
        # Handle both Python lists and numpy arrays
        if isinstance(actions, list):
            # Direct genny path - flatten list of lists to flat list
            flat_actions = []
            for action_pair in actions:
                flat_actions.extend([int(action_pair[0]), int(action_pair[1])])
            
            # Convert to SeqInt as required by genny bindings
            from tribal import SeqInt
            actions_seq = SeqInt()
            for action in flat_actions:
                actions_seq.append(action)
            
            # Send actions to the in-process environment controller (now using nimpy in same process!)
            import tribal
            actions_sent = tribal.set_external_actions_from_python(actions_seq)
            if not actions_sent:
                print("⚠️ Warning: Failed to send actions to ExternalNN controller")
            
            # Then call environment step - this will use the external controller in the same process
            print(f"🎯 Using in-process nimpy communication: step({len(flat_actions)} actions)")
            success = self._nim_env.step(actions_seq)
            if not success:
                raise RuntimeError("Direct genny step failed")
        else:
            # Fallback to numpy path through TribalGridEnv
            print(f"🔄 Using numpy fallback path")
            return self.tribal_env.step(actions)
        
        # Get results directly from genny bindings
        obs_data = self._nim_env.get_token_observations()
        observations = self.tribal_env._convert_observations(obs_data)
        
        rewards_seq = self._nim_env.get_rewards()
        rewards = [rewards_seq[i] for i in range(len(rewards_seq))]
        
        terminated_seq = self._nim_env.get_terminated()
        terminals = [terminated_seq[i] for i in range(len(terminated_seq))]
        
        truncated_seq = self._nim_env.get_truncated()
        truncations = [truncated_seq[i] for i in range(len(truncated_seq))]
        
        # Convert to numpy for compatibility
        import numpy as np
        rewards = np.array(rewards, dtype=np.float32)
        terminals = np.array(terminals, dtype=bool)
        truncations = np.array(truncations, dtype=bool)
        
        # Check for episode end
        if self._nim_env.is_episode_done():
            truncations[:] = True
        
        info = {
            "current_step": self._nim_env.get_current_step(),
            "max_steps": self.max_steps,
            "episode_done": self._nim_env.is_episode_done(),
        }
        
        return observations, rewards, terminals, truncations, info


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


class TribalProcessPlayTool(Tool):
    """
    Process-separated tribal environment tool.
    
    Uses file-based IPC to communicate with a Nim viewer process, eliminating
    the SIGSEGV issues caused by nimpy OpenGL conflicts.
    """
    
    env_config: TribalEnvConfig
    policy_uri: str | None = None
    
    def invoke(self, args: dict[str, str], overrides: list[str]) -> int | None:
        """Run tribal environment with process separation."""
        print("🎮 Tribal Process-Separated Play Tool")
        print("🎯 Using file-based IPC to eliminate SIGSEGV issues")
        
        # Import the process controller
        project_root = Path(__file__).parent.parent.parent
        tribal_dir = project_root / "tribal"
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        try:
            from tribal.src.tribal_process_controller import TribalProcessController
        except ImportError as e:
            print(f"❌ Failed to import process controller: {e}")
            return 1
        
        if self.policy_uri:
            # Support test modes
            if self.policy_uri in ["test_noop", "test_move"]:
                print(f"🧪 Test mode: {self.policy_uri}")
                return self._run_test_policy(TribalProcessController, tribal_dir)
            else:
                return self._run_with_neural_network(TribalProcessController, tribal_dir)
        else:
            return self._run_builtin_ai(TribalProcessController, tribal_dir)
    
    def _run_test_policy(self, ControllerClass, tribal_dir: Path) -> int:
        """Run with test policy (noop or move actions)."""
        print(f"🧪 Running test policy: {self.policy_uri}")
        
        try:
            with ControllerClass(tribal_dir) as controller:
                print("🚀 Starting Nim viewer process...")
                if not controller.start_nim_process():
                    print("❌ Failed to start Nim viewer process")
                    return 1
                
                print("📡 Activating communication...")
                controller.activate_communication()
                
                print("🔄 Resetting environment...")
                obs, info = controller.reset()
                print(f"✅ Environment reset - shape: {obs.shape}")
                
                # Run test loop for longer so you can see the viewer
                max_steps = min(500, self.env_config.game.max_steps)  # Increased from 100 to 500
                print(f"🎮 Running {max_steps} steps with {self.policy_uri}...")
                
                for step in range(max_steps):
                    # Generate test actions
                    actions = []
                    for agent in range(controller.num_agents):
                        if self.policy_uri == "test_noop":
                            actions.append([0, 0])  # NOOP action
                        elif self.policy_uri == "test_move":
                            import random
                            actions.append([1, random.randint(0, 3)])  # MOVE with random direction
                        else:
                            actions.append([0, 0])  # Default NOOP
                    
                    # Step environment
                    obs, rewards, terminals, truncations, info = controller.step(actions)
                    
                    # Log progress
                    if step % 20 == 0:
                        reward_sum = rewards.sum()
                        num_alive = (~(terminals | truncations)).sum()
                        print(f"  Step {step}: reward_sum={reward_sum:.3f}, agents_alive={num_alive}")
                        print(f"    Sample actions: agent_0={actions[0]}, agent_1={actions[1]}")
                    
                    if info.get("episode_done", False):
                        print(f"🏁 Episode ended at step {step}")
                        break
                    
                    # Add small delay so you can see the viewer window
                    import time
                    time.sleep(0.1)  # 100ms delay between steps
                
                print("🏆 SUCCESS: Process-separated communication working!")
                print("   Python controlled agents via file-based IPC")
                print("   Nim viewer displayed visualization without SIGSEGV")
                
                # Keep viewer open longer so you can see it
                print("💡 Keeping viewer open for 15 seconds...")
                print("   You should see the viewer window with moving agents!")
                print("   Press Ctrl+C to close early if needed.")
                time.sleep(15)
                
                return 0
                
        except Exception as e:
            print(f"❌ Error in test policy run: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    def _run_with_neural_network(self, ControllerClass, tribal_dir: Path) -> int:
        """Run with trained neural network policy."""
        try:
            print("🔄 Loading neural network...")
            from metta.rl.checkpoint_manager import CheckpointManager
            
            policy = CheckpointManager.load_from_uri(self.policy_uri)
            print(f"✅ Neural network loaded: {type(policy).__name__}")
            
            with ControllerClass(tribal_dir) as controller:
                print("🚀 Starting Nim viewer process...")
                if not controller.start_nim_process():
                    print("❌ Failed to start Nim viewer process")
                    return 1
                
                controller.activate_communication()
                
                obs, info = controller.reset()
                print(f"✅ Environment reset - shape: {obs.shape}")
                
                max_steps = min(200, self.env_config.game.max_steps)
                print(f"🎮 Running {max_steps} steps with neural network...")
                
                for step in range(max_steps):
                    # Get actions from policy
                    policy_output = policy.forward(obs)
                    actions = policy_output.get("actions")
                    
                    if actions is None:
                        print("⚠️  Policy didn't return actions, using noop")
                        actions = [[0, 0] for _ in range(controller.num_agents)]
                    
                    # Step environment
                    obs, rewards, terminals, truncations, info = controller.step(actions)
                    
                    # Log progress
                    if step % 50 == 0:
                        reward_sum = rewards.sum()
                        num_alive = (~(terminals | truncations)).sum()
                        print(f"  Step {step}: reward_sum={reward_sum:.3f}, agents_alive={num_alive}")
                    
                    if info.get("episode_done", False):
                        print(f"🏁 Episode ended at step {step}")
                        break
                
                print("🏆 SUCCESS: Neural network control with process separation!")
                time.sleep(3)
                return 0
                
        except Exception as e:
            print(f"❌ Error with neural network: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    def _run_builtin_ai(self, ControllerClass, tribal_dir: Path) -> int:
        """Run with built-in AI (Nim handles everything)."""
        print("🤖 Running with built-in AI (Nim controls agents)")
        
        try:
            with ControllerClass(tribal_dir) as controller:
                if not controller.start_nim_process():
                    print("❌ Failed to start Nim viewer process")
                    return 1
                
                controller.activate_communication()
                
                # Just keep the viewer running - Nim will handle AI internally
                print("🎮 Viewer running with built-in AI...")
                print("   Press Ctrl+C to stop")
                
                try:
                    while True:
                        time.sleep(1.0)
                except KeyboardInterrupt:
                    print("\n⏹️  Interrupted by user")
                    return 0
                
        except Exception as e:
            print(f"❌ Error with built-in AI: {e}")
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
    from tribal.src.tribal_genny import TribalGameConfig

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


def make_tribal_evals(
    base_env: TribalEnvConfig | None = None,
) -> list[SimulationConfig]:
    """Create evaluation suite for tribal environment with different scenarios."""
    base_env = base_env or make_tribal_environment()

    # Basic scenario - default settings
    basic_env = base_env.model_copy()

    # Combat-heavy scenario - increased Clippy spawns
    combat_env = base_env.model_copy()
    combat_env.game.clippy_spawn_rate = 1.0
    combat_env.game.clippy_damage = 2

    # Resource scarcity scenario
    scarcity_env = base_env.model_copy()
    scarcity_env.game.ore_per_battery = 3
    scarcity_env.game.batteries_per_heart = 3

    return [
        SimulationConfig(name="tribal/basic", env=basic_env, num_episodes=10),
        SimulationConfig(name="tribal/combat", env=combat_env, num_episodes=10),
        SimulationConfig(name="tribal/scarcity", env=scarcity_env, num_episodes=10),
    ]


def evaluate(
    policy_uri: str, simulations: list[SimulationConfig] | None = None
) -> SimTool:
    """
    Evaluate a trained policy on the tribal environment.

    Args:
        policy_uri: URI to trained policy (file:// or wandb://)
        simulations: Optional list of simulation configs, defaults to tribal evaluation suite
    """
    # Ensure tribal bindings are built
    _ensure_tribal_bindings_built()

    simulations = simulations or make_tribal_evals()

    return SimTool(
        simulations=simulations,
        policy_uris=[policy_uri],
    )


def play(
    env: TribalEnvConfig | None = None, policy_uri: str | None = None, **overrides
) -> "TribalProcessPlayTool":
    """
    Interactive play with the tribal environment using process separation.

    This uses the stable process-separated approach with our enhanced tribal_process_viewer
    that displays the full map with rectangle rendering to avoid SIGSEGV issues.

    Args:
        env: Optional tribal environment config
        policy_uri: Optional URI to trained policy (supports test_noop, test_move)
        **overrides: Additional configuration overrides
    """
    # Use the configured tribal environment, or create default
    play_env = env or make_tribal_environment()

    return TribalProcessPlayTool(
        env_config=play_env,
        policy_uri=policy_uri,
        **overrides,
    )



def play_process_separated(
    env: TribalEnvConfig | None = None, policy_uri: str | None = None, **overrides
) -> "TribalProcessPlayTool":
    """
    Interactive play with the tribal environment using process separation.

    This function is now identical to play() - both use the stable process-separated approach.

    Args:
        env: Optional tribal environment config
        policy_uri: Optional URI to trained policy (supports test_noop, test_move)
        **overrides: Additional configuration overrides
    """
    return play(env, policy_uri, **overrides)


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
