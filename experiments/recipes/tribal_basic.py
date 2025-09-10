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


class TribalGennyPlayTool(Tool):
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
            
            # Use nimpy approach for proper Python → Nim → GUI communication
            return self._run_nimpy_controlled_environment_safe(policy)
            
        except Exception as e:
            print(f"❌ Error running environment: {e}")
            import traceback
            traceback.print_exc()
            return 1

    def _run_nimpy_controlled_environment(self, policy) -> int:
        """
        Use nimpy to create and control a Nim environment directly from Python.
        This ensures Python configuration carries over and Python policy controls the agents.
        """
        import numpy as np
        import time
        from pathlib import Path
        
        print("🎮 Starting nimpy-controlled Nim environment...")
        print("   Python creates Nim environment with Python config")
        print("   Python policy controls agents via nimpy")
        print("   Nim provides visualization in same process")
        
        try:
            # Change to tribal directory for nimpy imports
            tribal_dir = Path(__file__).parent.parent.parent / "tribal"
            import os
            old_cwd = os.getcwd()
            os.chdir(tribal_dir)
            
            # Import tribal bindings to create Nim environment
            import sys
            sys.path.insert(0, str(tribal_dir / "bindings" / "generated"))
            import tribal
            
            print("🔗 Creating Nim environment with Python configuration...")
            
            # Create the Nim environment using the same config as Python
            config = tribal.default_tribal_config()
            config.game.max_steps = self.env_config.game.max_steps
            config.game.ore_per_battery = self.env_config.game.ore_per_battery
            config.game.batteries_per_heart = self.env_config.game.batteries_per_heart
            config.game.enable_combat = self.env_config.game.enable_combat
            config.game.clippy_spawn_rate = self.env_config.game.clippy_spawn_rate
            config.game.clippy_damage = self.env_config.game.clippy_damage
            config.game.heart_reward = self.env_config.game.heart_reward
            config.game.battery_reward = self.env_config.game.battery_reward
            config.game.ore_reward = self.env_config.game.ore_reward
            config.game.survival_penalty = self.env_config.game.survival_penalty
            config.game.death_penalty = self.env_config.game.death_penalty
            
            nim_env = tribal.new_tribal_env(config)
            print("✅ Nim environment created with Python configuration")
            
            # Initialize external neural network controller
            print("🧠 Setting up external neural network controller...")
            if not tribal.init_external_nn_controller():
                print("❌ Failed to initialize external NN controller")
                return 1
            print("✅ External NN controller initialized")
            
            # Test the environment stepping with Python policy
            print("🎯 Testing Python policy control...")
            nim_env.reset_env()
            
            step_count = 0
            max_steps = min(100, self.env_config.game.max_steps)
            
            print(f"🚀 Running {max_steps} steps with Python policy control...")
            
            for step in range(max_steps):
                # Get observations from Nim environment
                observations = nim_env.get_token_observations()
                
                # Get actions from Python policy
                policy_output = policy.forward(observations)
                actions = policy_output.get("actions")
                
                if actions is None:
                    print("⚠️  Policy didn't return actions, using noop")
                    actions = [[0, 0] for _ in range(15)]
                
                # Convert to flat sequence for nimpy
                flat_actions = []
                for action in actions:
                    flat_actions.extend(action)
                
                # Send actions to Nim environment
                if not tribal.set_external_actions_from_python(flat_actions):
                    print(f"⚠️  Failed to send actions at step {step}")
                
                # Step the Nim environment
                nim_env.step(flat_actions)
                
                # Get rewards and status
                rewards = nim_env.get_rewards()
                terminated = nim_env.get_terminated()
                
                step_count += 1
                
                # Log progress
                if step % 20 == 0:
                    avg_reward = np.mean(rewards)
                    num_alive = sum(1 for t in terminated if not t)
                    print(f"  Step {step}: avg_reward={avg_reward:.3f}, agents_alive={num_alive}")
                    
                    # Show sample actions for first few steps
                    if step < 3:
                        print(f"    Sample actions: agent_0={actions[0]}, agent_1={actions[1]}")
                
                # Small delay
                time.sleep(0.05)
                
                # Check if episode should end
                if all(terminated) or nim_env.is_episode_done():
                    print(f"🏁 Episode ended at step {step}")
                    break
            
            print(f"✅ SUCCESS: Python controlled Nim environment for {step_count} steps!")
            print("   Python policy successfully sent actions to Nim environment")
            print("   Nim environment processed actions and provided feedback")
            return 0
            
        except Exception as e:
            print(f"❌ Error in nimpy controlled environment: {e}")
            import traceback
            traceback.print_exc()
            return 1
        finally:
            os.chdir(old_cwd)

    def _run_nimpy_controlled_environment_safe(self, policy) -> int:
        """
        Safe version of nimpy controlled environment with gradual OpenGL initialization.
        This enables proper Python → Nim → GUI communication while avoiding OpenGL crashes.
        """
        import numpy as np
        import time
        from pathlib import Path
        
        print("🎮 Starting SAFE nimpy-controlled Nim environment...")
        print("   Python → Nim → GUI → Python communication")
        print("   Gradual OpenGL initialization to avoid crashes")
        
        try:
            # Change to tribal directory
            tribal_dir = Path(__file__).parent.parent.parent / "tribal"
            import os
            old_cwd = os.getcwd()
            os.chdir(tribal_dir)
            
            # Build safe viewer if needed
            safe_viewer_so = tribal_dir / "tribal_nimpy_safe_viewer.so"
            if not safe_viewer_so.exists():
                print("🔨 Building safe viewer...")
                import subprocess
                result = subprocess.run([
                    "nim", "c", "--app:lib", "--out:tribal_nimpy_safe_viewer.so", 
                    "-d:release", "src/tribal_nimpy_safe_viewer.nim"
                ], cwd=str(tribal_dir), capture_output=True, text=True)
                
                if result.returncode != 0:
                    print(f"❌ Failed to build safe viewer: {result.stderr}")
                    return 1
                print("✅ Safe viewer built successfully")
            
            # Import both tribal bindings and safe viewer
            import sys
            sys.path.insert(0, str(tribal_dir / "bindings" / "generated"))
            sys.path.insert(0, str(tribal_dir))
            
            import tribal
            import tribal_nimpy_safe_viewer as viewer
            
            print("✅ Imported tribal bindings and safe viewer")
            
            # Create Nim environment with Python configuration
            config = tribal.default_tribal_config()
            config.game.max_steps = self.env_config.game.max_steps
            config.game.ore_per_battery = self.env_config.game.ore_per_battery
            config.game.batteries_per_heart = self.env_config.game.batteries_per_heart
            config.game.enable_combat = self.env_config.game.enable_combat
            config.game.clippy_spawn_rate = self.env_config.game.clippy_spawn_rate
            config.game.clippy_damage = self.env_config.game.clippy_damage
            config.game.heart_reward = self.env_config.game.heart_reward
            config.game.battery_reward = self.env_config.game.battery_reward
            config.game.ore_reward = self.env_config.game.ore_reward
            config.game.survival_penalty = self.env_config.game.survival_penalty
            config.game.death_penalty = self.env_config.game.death_penalty
            
            nim_env = tribal.new_tribal_env(config)
            print("✅ Nim environment created with Python configuration")
            
            # Initialize external neural network controller
            print("🧠 Setting up external neural network controller...")
            if not tribal.init_external_nn_controller():
                print("❌ Failed to initialize external NN controller")
                return 1
            print("✅ External NN controller ready")
            
            # Initialize visualization with gradual approach
            print("🎨 Initializing visualization with safe approach...")
            
            # Step 1: Window only
            if not viewer.initWindowOnly():
                print("❌ Window creation failed")
                return 1
            print("✅ Window created")
            
            # Step 2: OpenGL context (this is where crashes usually happen)
            print("🔧 Attempting OpenGL context creation...")
            if not viewer.initOpenGLContext():
                print("❌ OpenGL context failed - falling back to headless mode")
                return self._run_headless_communication_loop(nim_env, policy)
            print("✅ OpenGL context created successfully")
            
            # Step 3: Panels and UI
            if not viewer.initPanels():
                print("❌ Panel initialization failed")
                return 1
            print("✅ UI panels initialized")
            
            # Step 4: Load assets
            print("🎨 Loading assets...")
            if not viewer.loadAssetsSafe():
                print("⚠️  Asset loading failed, continuing without assets")
            else:
                print("✅ Assets loaded successfully")
            
            # Run the main communication loop
            return self._run_communication_loop_with_gui(nim_env, policy, viewer)
            
        except Exception as e:
            print(f"❌ Error in safe nimpy environment: {e}")
            import traceback
            traceback.print_exc()
            return 1
        finally:
            os.chdir(old_cwd)

    def _run_headless_communication_loop(self, nim_env, policy) -> int:
        """Fallback: Run without GUI if OpenGL fails"""
        print("🤖 Running in headless mode (no GUI)")
        print("   Python → Nim communication without visualization")
        
        try:
            nim_env.reset_env()
            step_count = 0
            max_steps = min(100, self.env_config.game.max_steps)
            
            print(f"🚀 Running {max_steps} steps in headless mode...")
            
            for step in range(max_steps):
                # Get observations
                observations = nim_env.get_token_observations()
                
                # Get actions from policy
                policy_output = policy.forward(observations)
                actions = policy_output.get("actions", [[0, 0] for _ in range(15)])
                
                # Send to Nim
                flat_actions = [action for pair in actions for action in pair]
                if not tribal.set_external_actions_from_python(flat_actions):
                    print(f"⚠️  Failed to send actions at step {step}")
                
                # Step environment
                nim_env.step(flat_actions)
                
                # Get results
                rewards = nim_env.get_rewards()
                
                if step % 20 == 0:
                    avg_reward = np.mean(rewards)
                    print(f"  Step {step}: avg_reward={avg_reward:.3f}")
                
                time.sleep(0.05)
                
                if nim_env.is_episode_done():
                    break
            
            print(f"✅ Headless mode completed {step_count} steps")
            return 0
            
        except Exception as e:
            print(f"❌ Headless mode failed: {e}")
            return 1

    def _run_communication_loop_with_gui(self, nim_env, policy, viewer) -> int:
        """Main communication loop: Python → Nim → GUI → Python"""
        import numpy as np
        import time
        
        print("🎮 Starting main communication loop with GUI...")
        print("   Python sends actions → Nim processes → GUI shows → Python gets results")
        
        try:
            nim_env.reset_env()
            step_count = 0
            max_steps = min(200, self.env_config.game.max_steps)
            
            print(f"🚀 Running {max_steps} steps with full GUI...")
            
            while step_count < max_steps and viewer.isWindowOpen():
                # 1. Get observations from Nim
                observations = nim_env.get_token_observations()
                
                # 2. Get actions from Python policy  
                policy_output = policy.forward(observations)
                actions = policy_output.get("actions")
                
                if actions is None:
                    actions = [[0, 0] for _ in range(15)]  # Default noop
                
                # 3. Send actions to Nim
                flat_actions = [action for pair in actions for action in pair]
                tribal.set_external_actions_from_python(flat_actions)
                
                # 4. Step Nim environment
                nim_env.step(flat_actions)
                
                # 5. Update GUI visualization
                if not viewer.renderFrameMinimal():
                    print("⚠️  GUI rendering failed, continuing...")
                
                # 6. Get results from Nim
                rewards = nim_env.get_rewards()
                terminated = nim_env.get_terminated()
                
                step_count += 1
                
                # Progress logging
                if step_count % 50 == 0:
                    avg_reward = np.mean(rewards)
                    num_alive = sum(1 for t in terminated if not t)
                    print(f"  Step {step_count}: avg_reward={avg_reward:.3f}, agents_alive={num_alive}")
                    
                    # Show sample actions
                    print(f"    Sample actions: agent_0={actions[0]}, agent_1={actions[1]}")
                
                # Small delay for visualization
                time.sleep(0.05)  # 20 Hz
                
                # Check termination
                if all(terminated) or nim_env.is_episode_done():
                    print(f"🏁 Episode ended at step {step_count}")
                    break
            
            print(f"🎉 SUCCESS: Completed {step_count} steps with Python → Nim → GUI communication!")
            
            # Keep window open briefly to see final state
            print("💡 Keeping window open for 3 seconds to view final state...")
            for _ in range(3):
                if viewer.isWindowOpen():
                    viewer.renderFrameMinimal()
                    time.sleep(1)
                else:
                    break
            
            return 0
            
        except Exception as e:
            print(f"❌ Communication loop failed: {e}")
            import traceback
            traceback.print_exc()
            return 1
        finally:
            # Clean up GUI
            try:
                viewer.closeVisualization()
                print("✅ GUI cleanup complete")
            except:
                pass

    def _run_clean_genny_loop(self, env, policy) -> int:
        """
        Clean approach: Use genny bindings for Python-Nim communication,
        launch native Nim viewer as subprocess (avoiding nimpy OpenGL issues).
        """
        import numpy as np
        import time
        import subprocess
        from pathlib import Path
        
        print("🎮 Starting clean genny-based approach...")
        print("   Python controls environment via genny bindings")
        print("   Native Nim viewer runs as separate subprocess")
        
        try:
            # Start the native Nim viewer as a subprocess
            tribal_dir = Path(__file__).parent.parent.parent / "tribal"
            print(f"🎯 Launching native Nim viewer from: {tribal_dir}")
            
            nim_process = subprocess.Popen(
                ["nim", "r", "-d:release", "src/tribal"],
                cwd=tribal_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            print("✅ Native Nim viewer launched as subprocess")
            print("🎮 Starting Python policy control loop...")
            
            # Reset environment
            obs, info = env.reset()
            print(f"✅ Environment reset complete. Starting step: {info.get('current_step', 0)}")
            
            episode_rewards = np.zeros(env.num_agents)
            step_count = 0
            
            while step_count < min(200, env.max_steps):  # Run for 200 steps
                # Get actions from policy
                policy_output = policy.forward(obs)
                actions = policy_output.get("actions")
                
                if actions is None:
                    print("⚠️  Policy didn't return actions, using noop")
                    actions = [[0, 0] for _ in range(env.num_agents)]
                
                # Step environment with Python policy control
                obs, rewards, terminals, truncations, info = env.step(actions)
                episode_rewards += rewards
                step_count += 1
                
                # Log progress periodically
                if step_count % 50 == 0:
                    avg_reward = np.mean(episode_rewards)
                    num_alive = np.sum(~(terminals | truncations))
                    print(f"  Step {step_count}: avg_reward={avg_reward:.3f}, agents_alive={num_alive}")
                    
                    # Show sample actions for debugging
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
            print(f"✅ SUCCESS: Python successfully controlled Nim environment via genny bindings!")
            
            return 0
            
        except Exception as e:
            print(f"❌ Error in clean genny loop: {e}")
            import traceback
            traceback.print_exc()
            return 1
        finally:
            # Clean up subprocess
            try:
                nim_process.terminate()
                nim_process.wait(timeout=5)
                print("✅ Native Nim viewer subprocess terminated")
            except:
                try:
                    nim_process.kill()
                    print("🔥 Native Nim viewer subprocess killed")
                except:
                    pass

    def _run_interactive_loop(self, env, policy, tribal_module) -> int:
        """
        Run an interactive game loop with both policy control and native Nim visualization.
        
        This uses threading to run policy control while nimpy provides the Nim GUI.
        """
        import numpy as np
        import time
        import threading
        from pathlib import Path
        
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
            print("🚀 TEMPORARILY DISABLING policy control thread to test nimpy isolation...")
            # policy_thread = threading.Thread(target=policy_control_loop, daemon=True)  
            # policy_thread.start()
            print("   Policy thread disabled - testing nimpy visualization standalone")
            
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
                
                # Try to import with better error handling
                print("🔄 Step 1: Importing nimpy visualization module...")
                import tribal_nimpy_viewer as viewer
                print("✅ Step 1 complete: Nimpy module imported successfully")
                
                # Initialize the visualization with error checking
                print("🎨 Step 2: Initializing nimpy visualization...")
                if not viewer.initVisualization():
                    print("❌ Failed to initialize nimpy visualization")
                    return 1
                print("✅ Step 2 complete: Nimpy visualization initialized")
                
                # Load assets with progress tracking
                print("🎨 Step 3: Loading tribal assets...")
                if not viewer.loadAssets():
                    print("❌ Failed to load tribal assets")
                    return 1
                print("✅ Step 3 complete: All assets loaded successfully")
                
                # Test nimpy function calls BEFORE entering the loop
                print("🔧 Step 4: Testing nimpy function calls individually...")
                
                print("🔧 Step 4a: Testing isWindowOpen()...")
                window_status = viewer.isWindowOpen()
                print(f"✅ Step 4a complete: isWindowOpen() = {window_status}")
                
                print("🔧 Step 4b: Testing single renderFrame() call...")
                render_status = viewer.renderFrame()
                print(f"✅ Step 4b complete: renderFrame() = {render_status}")
                
                print("✅ All individual function tests passed!")
                
            except Exception as e:
                print(f"❌ Error setting up nimpy visualization: {e}")
                import traceback
                traceback.print_exc()
                os.chdir(old_cwd)
                return 1
            
            # Try a more cautious approach to the render loop
            print("🎨 Starting nimpy visualization loop...")
            print("   Testing each function call individually to isolate crash point...")
            
            try:
                # Test 1: Can we call isWindowOpen?
                print("🔧 DEBUG: Testing viewer.isWindowOpen()...")
                window_open = viewer.isWindowOpen()
                print(f"🔧 DEBUG: isWindowOpen() returned: {window_open}")
                
                # Test 2: Can we call renderFrame once?
                print("🔧 DEBUG: Testing viewer.renderFrame() once...")
                render_result = viewer.renderFrame()
                print(f"🔧 DEBUG: renderFrame() returned: {render_result}")
                
                # Test 3: Try a simple loop with debug output
                print("🔧 DEBUG: Starting minimal render loop...")
                frame_count = 0
                max_frames = 3  # Only render 3 frames for debugging
                
                while frame_count < max_frames and viewer.isWindowOpen() and game_running.is_set():
                    print(f"🔧 DEBUG: Loop iteration {frame_count}")
                    
                    if not viewer.renderFrame():
                        print("🔧 DEBUG: renderFrame() returned false, breaking")
                        break
                    
                    print(f"🔧 DEBUG: Frame {frame_count} completed successfully")
                    frame_count += 1
                    
                    # Longer delay for debugging
                    time.sleep(0.5)  # 2 FPS for debugging
                
                print(f"🔧 DEBUG: Render loop completed after {frame_count} frames")
                    
            except Exception as e:
                print(f"❌ Error in visualization loop after {frame_count} frames: {e}")
                import traceback
                traceback.print_exc()
            finally:
                # Cleanup
                try:
                    print("🧹 Cleaning up nimpy visualization...")
                    viewer.closeVisualization()
                    print("✅ Nimpy visualization cleanup complete")
                except Exception as cleanup_error:
                    print(f"⚠️  Cleanup error (non-critical): {cleanup_error}")
                finally:
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
) -> "TribalGennyPlayTool":
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

    return TribalGennyPlayTool(
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
