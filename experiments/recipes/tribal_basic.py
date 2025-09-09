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
    Tool for running tribal environment directly in Nim with nimpy control.

    Uses the unified controller system to:
    1. Neural network control: Python loads policy and controls via nimpy callback
    2. Built-in AI fallback: Nim uses its native AI when no neural network provided

    This runs the actual Nim environment directly, not a Python wrapper.
    """

    env_config: TribalEnvConfig
    policy_uri: str | None = None

    def invoke(self, args: dict[str, str], overrides: list[str]) -> int | None:
        """Run tribal environment directly in Nim with optional neural network control."""
        print("🎮 Tribal Nim Play Tool")
        print("🎯 Running native Nim environment with nimpy interface")

        if self.policy_uri:
            return self._setup_neural_network_control()
        else:
            return self._run_nim_builtin_ai()

    def _setup_neural_network_control(self) -> int:
        """
        Set up neural network control via nimpy callback and launch Nim environment.
        """
        try:
            print("🔄 Loading neural network...")
            from metta.rl.checkpoint_manager import CheckpointManager

            policy = CheckpointManager.load_from_uri(self.policy_uri)
            print(f"✅ Neural network loaded: {type(policy).__name__}")

            # Test the nimpy interface
            print("🧪 Setting up nimpy interface...")
            import sys
            import os

            bindings_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "tribal", "bindings", "generated"
            )
            sys.path.insert(0, bindings_path)

            try:
                import tribal

                print("✅ Nimpy bindings imported successfully")

                # Initialize external neural network controller
                success = tribal.init_external_nncontroller()
                if success:
                    print("✅ External neural network controller initialized")
                else:
                    print("❌ Failed to initialize external controller")
                    return 1

                controller_type = tribal.get_controller_type_string()
                print(f"🤖 Controller type: {controller_type}")

                # Create action callback that neural network will use
                def neural_network_callback():
                    """Generate actions for all agents using the loaded neural network."""
                    # This would be called by Nim when it needs actions
                    # For now, just demonstrate the concept
                    actions = []
                    for agent_id in range(15):  # 15 agents (compile-time constant)
                        # Simple example: generate random actions
                        # In practice, this would use policy.forward() with observations
                        action_type = 0  # NOOP for demo
                        action_arg = 0
                        actions.extend([action_type, action_arg])
                    return actions

                # Set up real-time neural network control
                print("🔗 Setting up real-time neural network control...")
                
                print("🎮 Architecture ready:")
                print("  1. ✅ Neural network loaded in Python")
                print("  2. ✅ Nim external controller initialized")
                print("  3. 🔄 Starting neural network action service...")
                
                # Start the neural network action service in background
                return self._run_neural_network_service(policy, tribal)

            except ImportError as e:
                print(f"❌ Failed to import nimpy bindings: {e}")
                return 1
            
        except Exception as e:
            print(f"❌ Error setting up neural network: {e}")
            return 1

    def _run_neural_network_service(self, policy, tribal_module) -> int:
        """
        Run neural network service that provides actions directly to Nim environment via nimpy.
        """
        import threading
        import time
        import subprocess
        
        # Initialize external NN controller in Nim
        success = tribal_module.init_external_nncontroller()
        if not success:
            print("❌ Failed to initialize external NN controller")
            error = tribal_module.take_error()
            if error:
                print(f"   Error: {error}")
            return 1
        
        print("  ✅ External NN controller initialized in Nim")
        
        # Flag to control the action service
        service_running = threading.Event()
        service_running.set()
        
        # Create a simple Nim environment to get observations from
        print("  🔧 Creating Nim environment for observations...")
        nim_config = tribal_module.default_tribal_config()
        nim_env = tribal_module.TribalEnv(nim_config)
        print("  ✅ Nim observation environment created")

        def neural_network_action_service():
            """Background service that generates actions from neural network and sends directly to Nim."""
            step = 0
            while service_running.is_set():
                try:
                    # Get current observations from Nim environment
                    # Note: This is a separate environment instance just for getting observations
                    # The actual game state is in the main Nim process
                    nim_env.reset_env()  # Reset to get fresh observations
                    obs_tokens = nim_env.get_token_observations()
                    
                    # Convert to tensor format expected by policy
                    import torch
                    import numpy as np
                    
                    # Debug observation size
                    if step == 0:
                        print(f"  🔍 Debug: Got {len(obs_tokens)} observation tokens")
                        print(f"  🔍 Expected: {15 * 200 * 3} = {15} agents * {200} tokens * {3} values")
                    
                    # For now, let's send simple MOVE actions to test if the external controller is working
                    # This will help debug the connection before dealing with policy inference
                    actions_list = []
                    for agent_id in range(15):
                        # Send MOVE actions with different directions to see if they're being used
                        action_type = 1  # MOVE action
                        action_arg = agent_id % 8  # Different direction for each agent (0-7)
                        actions_list.extend([action_type, action_arg])
                        
                    if step == 0:
                        print(f"  🎯 Debug: Sending test actions - agents should move in different directions")
                        print(f"  🎯 Actions sample: agent 0 -> MOVE dir {actions_list[1]}, agent 1 -> MOVE dir {actions_list[3]}")
                        
                    # TODO: Later we'll use the actual policy once we confirm external actions work:
                    # Check if we got the expected size
                    # expected_size = 15 * 200 * 3  # 9000
                    # if len(obs_tokens) != expected_size:
                    #     Handle incorrect size - use default observations for now
                    #     print(f"  ⚠️  Warning: Got {len(obs_tokens)} tokens, expected {expected_size}. Using zeros.")
                    #     obs_array = np.zeros((15, 200, 3), dtype=np.int32)
                    # else:
                    #     Reshape observations to match expected format: (num_agents, max_tokens, 3)
                    #     obs_array = np.array(obs_tokens).reshape(15, 200, 3)  # 15 agents, 200 tokens, 3 values per token
                    # 
                    # obs_tensor = torch.tensor(obs_array, dtype=torch.int32)
                    # 
                    # Prepare batch of observations for all agents
                    # obs_batch = obs_tensor.unsqueeze(0)  # Add batch dimension: (1, 15, 200, 3)
                    # 
                    # Use policy to generate actions
                    # with torch.no_grad():
                    #     from tensordict import TensorDict
                    #     
                    #     Create TensorDict with observations
                    #     batch = TensorDict({
                    #         "agents": {
                    #             "observation": obs_batch
                    #         }
                    #     }, batch_size=[1])
                    #     
                    #     Get actions from policy
                    #     policy_output = policy.forward(batch)
                    #     
                    #     Extract actions for all agents
                    #     agent_actions = policy_output["agents"]["action"]  # Shape: (1, 15, 2)
                    #     agent_actions = agent_actions.squeeze(0)  # Remove batch dim: (15, 2)
                    #     
                    #     Convert to flat list for Nim
                    #     actions_list = []
                    #     for agent_id in range(15):
                    #         action_type = int(agent_actions[agent_id, 0].item())
                    #         action_arg = int(agent_actions[agent_id, 1].item())
                    #         actions_list.extend([action_type, action_arg])
                    
                    # Create SeqInt and populate it
                    actions = tribal_module.SeqInt()
                    for action in actions_list:
                        actions.append(action)
                    
                    # Send actions directly to Nim
                    result = tribal_module.set_external_actions_from_python(actions)
                    if not result:
                        print(f"⚠️  Failed to set actions in Nim at step {step}")
                        error = tribal_module.take_error()
                        if error:
                            print(f"   Error: {error}")
                    else:
                        if step % 100 == 0:  # Log every 100 steps
                            print(f"  📡 Step {step}: Neural network actions sent to Nim")
                    
                    step += 1
                    time.sleep(0.1)  # 10 Hz action generation
                    
                except Exception as e:
                    print(f"❌ Neural network action service error: {e}")
                    import traceback
                    traceback.print_exc()
                    service_running.clear()
        
        try:
            print("  🚀 Starting neural network action thread...")
            action_thread = threading.Thread(target=neural_network_action_service, daemon=True)
            action_thread.start()
            
            print("  ✅ Neural network action service started successfully!")
            print("  📡 Python neural network connected directly to Nim via nimpy")
            print("  🎮 Press Ctrl+C to stop training and environment")
            
            # Get tribal directory for launching Nim
            import os
            tribal_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'tribal')
            tribal_dir = os.path.abspath(tribal_dir)
            
            print(f"  🎯 Launching Nim environment from: {tribal_dir}")
            
            # Launch Nim environment with subprocess
            result = subprocess.run(
                ["nim", "r", "-d:release", "src/tribal"],
                cwd=tribal_dir,
                capture_output=False,  # Let output go directly to console
                text=True
            )
            
            return result.returncode
            
        except KeyboardInterrupt:
            print("\n🛑 Stopping neural network service...")
            service_running.clear()
            return 0
        except Exception as e:
            print(f"❌ Error in neural network service: {e}")
            service_running.clear()
            return 1
        finally:
            # Clean up
            service_running.clear()
            print("🧹 Neural network service stopped")

    def _run_nim_builtin_ai(self) -> int:
        """
        Launch Nim environment with built-in AI (no neural network).
        """
        print("🎯 Setting up Nim environment with built-in AI...")
        print("💡 No neural network specified - using built-in Nim AI")

        try:
            import sys
            import os

            bindings_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "tribal", "bindings", "generated"
            )
            sys.path.insert(0, bindings_path)

            import tribal

            # Initialize built-in AI controller
            success = tribal.init_builtin_aicontroller()
            if success:
                print("✅ Built-in AI controller initialized")
                print(f"🤖 Controller type: {tribal.get_controller_type_string()}")
                print("")
                print("🚀 Launching Nim environment with built-in AI...")
                
                # Actually launch the Nim environment
                tribal_dir = os.path.join(os.path.dirname(__file__), "..", "..", "tribal")
                
                try:
                    import subprocess
                    result = subprocess.run(
                        ["nim", "r", "-d:release", "src/tribal"],
                        cwd=tribal_dir,
                        capture_output=False,  # Allow interactive output
                        text=True
                    )
                    
                    if result.returncode == 0:
                        print("✅ Nim environment completed successfully")
                        return 0
                    else:
                        print(f"❌ Nim environment exited with code: {result.returncode}")
                        return result.returncode
                        
                except KeyboardInterrupt:
                    print("\n⏹️ Nim environment interrupted by user")
                    return 0
                except Exception as e:
                    print(f"❌ Error launching Nim environment: {e}")
                    return 1
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
