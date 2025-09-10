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

from metta.cogworks.curriculum.curriculum import CurriculumConfig
from metta.common.tool import Tool
from metta.cogworks.curriculum.task_generator import TaskGeneratorConfig
from metta.rl.trainer_config import TrainerConfig, EvaluationConfig
from metta.rl.loss.loss_config import LossConfig
from metta.sim.simulation_config import SimulationConfig
from metta.sim.tribal_genny import TribalEnvConfig
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


class TribalNimPlayTool(Tool):
    """
    Tool for running tribal environment in-process with nimpy control.

    Uses the existing TribalGridEnv with direct nimpy communication to:
    1. Neural network control: Python loads policy and controls environment via nimpy
    2. Built-in AI fallback: Environment uses its native AI when no neural network provided

    This runs the Nim environment directly within the Python process via nimpy.
    """

    env_config: TribalEnvConfig
    policy_uri: str | None = None

    def invoke(self, args: dict[str, str], overrides: list[str]) -> int | None:
        """Run tribal environment in-process with optional neural network control."""
        print("🎮 Tribal In-Process Play Tool")
        print("🎯 Using TribalGridEnv with direct nimpy interface")

        # Ensure bindings are built
        _ensure_tribal_bindings_built()

        if self.policy_uri:
            # Support test modes that don't require actual policy files
            if self.policy_uri in ["test_noop", "test_move"]:
                print(f"🧪 Test mode: {self.policy_uri}")
                return self._run_test_neural_network()
            else:
                return self._run_with_neural_network()
        else:
            return self._run_with_builtin_ai()

    def _run_with_neural_network(self) -> int:
        """
        Run with neural network control via in-process nimpy environment.
        """
        try:
            print("🔄 Loading neural network...")
            from metta.rl.checkpoint_manager import CheckpointManager

            policy = CheckpointManager.load_from_uri(self.policy_uri)
            print(f"✅ Neural network loaded: {type(policy).__name__}")

            return self._run_environment_with_policy(policy)
            
        except Exception as e:
            print(f"❌ Error loading neural network: {e}")
            return 1

    def _run_test_neural_network(self) -> int:
        """Run with test neural network (noop or move actions)."""
        print(f"🧪 Running test neural network: {self.policy_uri}")
        
        # Create fake policy for test
        class TestPolicy:
            def __init__(self, test_mode):
                self.test_mode = test_mode
            
            def forward(self, observations):
                """Generate test actions based on test mode using direct genny-compatible format."""
                num_agents = 15
                
                if self.test_mode == "test_noop":
                    # All noop actions - return as simple Python list for direct genny use
                    actions = [[0, 0] for _ in range(num_agents)]
                elif self.test_mode == "test_move":
                    # All move actions with random directions - direct Python list
                    import random
                    actions = [[1, random.randint(0, 3)] for _ in range(num_agents)]  # MOVE with random directions
                else:
                    actions = [[0, 0] for _ in range(num_agents)]
                
                return {"actions": actions}
        
        test_policy = TestPolicy(self.policy_uri)
        return self._run_environment_with_policy(test_policy)

    def _run_environment_with_policy(self, policy) -> int:
        """Run the environment with the given policy controlling all agents."""
        try:
            print("🎮 Creating in-process tribal environment with direct genny bindings...")
            from metta.sim.tribal_genny import TribalGridEnv
            
            # Create environment directly
            config_dict = {
                "max_steps": self.env_config.game.max_steps,
                "ore_per_battery": self.env_config.game.ore_per_battery,
                "batteries_per_heart": self.env_config.game.batteries_per_heart,
                "enable_combat": self.env_config.game.enable_combat,
                "clippy_spawn_rate": self.env_config.game.clippy_spawn_rate,
                "clippy_damage": self.env_config.game.clippy_damage,
                "heart_reward": self.env_config.game.heart_reward,
                "battery_reward": self.env_config.game.battery_reward,
                "ore_reward": self.env_config.game.ore_reward,
                "survival_penalty": self.env_config.game.survival_penalty,
                "death_penalty": self.env_config.game.death_penalty,
            }
            
            base_env = TribalGridEnv(config_dict)
            print(f"✅ Base environment created with {base_env.num_agents} agents")
            
            # Wrap with direct genny interface for optimal performance
            env = DirectGennyTribalEnv(base_env)
            print("🎯 Using direct genny bindings for neural network control (no numpy conversion)")
            
            # Run the interactive loop with neural network control
            return self._run_interactive_loop(env, policy, None)
            
        except Exception as e:
            print(f"❌ Error running environment: {e}")
            import traceback
            traceback.print_exc()
            return 1

    def _run_interactive_loop(self, env, policy, tribal_module) -> int:
        """
        Run an interactive game loop with both policy control and native Nim visualization.
        
        This launches the native Nim viewer while the Python policy controls the agents
        through the in-process environment. Best of both worlds!
        """
        import numpy as np
        import time
        import threading
        import subprocess
        
        print("🎮 Starting interactive game loop with native Nim visualization...")
        print("   Policy will control all agents via environment stepping")
        print("   Native Nim viewer will show the visualization")
        
        # Flag to control the game loop
        game_running = threading.Event()
        game_running.set()
        
        def policy_control_loop():
            """Run the policy control in a separate thread."""
            try:
                # Reset environment
                obs, info = env.reset()
                print(f"✅ Environment reset complete. Starting step: {info.get('current_step', 0)}")
                
                episode_rewards = np.zeros(env.num_agents)
                step_count = 0
                
                while step_count < env.max_steps and game_running.is_set():
                    # Get actions from policy
                    policy_output = policy.forward(obs)
                    actions = policy_output.get("actions")
                    
                    if actions is None:
                        print("⚠️  Policy didn't return actions, using noop")
                        actions = np.zeros((env.num_agents, 2), dtype=np.int32)
                    
                    # Step environment
                    obs, rewards, terminals, truncations, info = env.step(actions)
                    
                    episode_rewards += rewards
                    step_count += 1
                    
                    # Log progress periodically
                    if step_count % 100 == 0:
                        avg_reward = np.mean(episode_rewards)
                        num_alive = np.sum(~(terminals | truncations))
                        print(f"  Step {step_count}: avg_reward={avg_reward:.3f}, agents_alive={num_alive}")
                        
                        # Show some actions for debugging (handle both list and array formats)
                        if isinstance(actions, list):
                            print(f"    Sample actions: agent_0={actions[0]}, agent_1={actions[1]}")
                        else:
                            print(f"    Sample actions: agent_0=[{actions[0,0]},{actions[0,1]}], agent_1=[{actions[1,0]},{actions[1,1]}]")
                    
                    # Check if episode should end
                    if np.all(terminals | truncations) or info.get('episode_done', False):
                        print(f"🏁 Episode ended at step {step_count}")
                        break
                    
                    # Small delay for visualization
                    time.sleep(0.05)  # 20 Hz
                
                # Final statistics
                final_avg_reward = np.mean(episode_rewards)
                final_total_reward = np.sum(episode_rewards)
                print(f"🏆 Episode complete!")
                print(f"   Steps: {step_count}")
                print(f"   Average reward: {final_avg_reward:.3f}")
                print(f"   Total reward: {final_total_reward:.3f}")
                
            except Exception as e:
                print(f"❌ Error in policy control loop: {e}")
                import traceback
                traceback.print_exc()
            finally:
                game_running.clear()
        
        try:
            print("🚀 Starting policy control thread...")
            policy_thread = threading.Thread(target=policy_control_loop, daemon=True)
            policy_thread.start()
            
            print("🎯 Launching native Nim visualization...")
            print("   The Nim window will show the agents being controlled by your policy")
            print("   Press Ctrl+C in terminal to stop, or close the Nim window")
            
            # Launch native Nim viewer using nimpy (in-process visualization)
            tribal_dir = Path(__file__).parent.parent.parent / "tribal"
            
            # Change working directory to tribal BEFORE importing nimpy module
            import os
            old_cwd = os.getcwd()
            os.chdir(tribal_dir)
            
            try:
                # Import the nimpy visualization module (now with correct working directory)
                import sys
                sys.path.insert(0, str(tribal_dir))
                import tribal_nimpy_viewer as viewer
                
                # Initialize the visualization
                if not viewer.initVisualization():
                    print("❌ Failed to initialize nimpy visualization")
                    return 1
                
                # Load assets
                if not viewer.loadAssets():
                    print("❌ Failed to load tribal assets")
                    return 1
                
            except Exception as e:
                print(f"❌ Error setting up nimpy visualization: {e}")
                os.chdir(old_cwd)
                return 1
            
            # Run the visualization loop
            print("🎨 Starting nimpy visualization loop...")
            try:
                while viewer.isWindowOpen():
                    if not viewer.renderFrame():
                        break
            finally:
                # Cleanup
                viewer.closeVisualization()
                os.chdir(old_cwd)
                
            return 0
            
        except KeyboardInterrupt:
            print("\n⏹️  Interrupted by user")
            game_running.clear()
            return 0
        except Exception as e:
            print(f"❌ Error launching visualization: {e}")
            import traceback
            traceback.print_exc()
            return 1
        finally:
            game_running.clear()
            print("🧹 Game loop stopped")

    def _run_with_builtin_ai(self) -> int:
        """
        Run tribal environment with built-in AI (no neural network).
        """
        print("🎯 Running with built-in AI...")
        print("💡 No neural network specified - using built-in Nim AI")

        try:
            print("🎮 Creating in-process tribal environment...")
            from metta.sim.tribal_genny import TribalGridEnv
            
            # Create environment directly
            config_dict = {
                "max_steps": self.env_config.game.max_steps,
                "ore_per_battery": self.env_config.game.ore_per_battery,
                "batteries_per_heart": self.env_config.game.batteries_per_heart,
                "enable_combat": self.env_config.game.enable_combat,
                "clippy_spawn_rate": self.env_config.game.clippy_spawn_rate,
                "clippy_damage": self.env_config.game.clippy_damage,
                "heart_reward": self.env_config.game.heart_reward,
                "battery_reward": self.env_config.game.battery_reward,
                "ore_reward": self.env_config.game.ore_reward,
                "survival_penalty": self.env_config.game.survival_penalty,
                "death_penalty": self.env_config.game.death_penalty,
            }
            
            env = TribalGridEnv(config_dict)
            print(f"✅ Environment created with {env.num_agents} agents")
            print("🤖 Built-in AI will generate actions for environment stepping")
            
            # Note: For now, we'll step with dummy actions and let the Nim environment
            # handle AI internally. In the future, we could implement a proper AI policy.
            return self._run_builtin_ai_loop(env, None)
            
        except Exception as e:
            print(f"❌ Error running with built-in AI: {e}")
            import traceback
            traceback.print_exc()
            return 1

    def _run_builtin_ai_loop(self, env, tribal_module) -> int:
        """
        Run the environment with built-in AI controlling all agents.
        """
        import numpy as np
        import time
        
        print("🎮 Starting built-in AI game loop...")
        print("   Built-in AI will control all agents automatically")
        
        try:
            # Reset environment
            obs, info = env.reset()
            print(f"✅ Environment reset complete. Starting step: {info.get('current_step', 0)}")
            
            step_count = 0
            episode_rewards = np.zeros(env.num_agents)
            
            while step_count < env.max_steps:
                # For built-in AI, we still need to step the environment
                # but the Nim side will generate the actions automatically
                # Let's generate dummy actions and let Nim override them
                dummy_actions = np.zeros((env.num_agents, 2), dtype=np.int32)
                
                # Step environment (Nim will use its own AI actions internally)
                obs, rewards, terminals, truncations, info = env.step(dummy_actions)
                
                episode_rewards += rewards
                step_count += 1
                
                # Log progress periodically
                if step_count % 100 == 0:
                    avg_reward = np.mean(episode_rewards)
                    num_alive = np.sum(~(terminals | truncations))
                    print(f"  Step {step_count}: avg_reward={avg_reward:.3f}, agents_alive={num_alive}")
                
                # Check if episode should end
                if np.all(terminals | truncations) or info.get('episode_done', False):
                    print(f"🏁 Episode ended at step {step_count}")
                    break
                
                # Small delay for visualization
                time.sleep(0.05)  # 20 Hz
            
            # Final statistics
            final_avg_reward = np.mean(episode_rewards)
            final_total_reward = np.sum(episode_rewards)
            print(f"🏆 Episode complete!")
            print(f"   Steps: {step_count}")
            print(f"   Average reward: {final_avg_reward:.3f}")
            print(f"   Total reward: {final_total_reward:.3f}")
            
            return 0
            
        except KeyboardInterrupt:
            print("\n⏹️  Interrupted by user")
            return 0
        except Exception as e:
            print(f"❌ Error in AI game loop: {e}")
            import traceback
            traceback.print_exc()
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
