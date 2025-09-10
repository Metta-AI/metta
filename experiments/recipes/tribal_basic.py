"""
Tribal environment recipes using pure genny bindings.

This provides a clean implementation that uses the genny bindings directly
without the complex nimpy visualization approach that was causing crashes.
"""

from pathlib import Path
from typing import Any

from metta.core.config_parsing import Tool
from metta.core.simulation import SimTool, ReplayTool
from metta.rl.env.env_config import EnvConfig, EnvType
from metta.rl.simulation.simulation_config import SimulationConfig


class TribalEnvGameConfig:
    """Configuration for tribal game parameters."""
    
    def __init__(
        self,
        max_steps: int = 1000,
        ore_per_battery: int = 3,
        batteries_per_heart: int = 2,
        enable_combat: bool = True,
        clippy_spawn_rate: float = 1.0,
        clippy_damage: int = 1,
        heart_reward: float = 1.0,
        ore_reward: float = 0.1,
        battery_reward: float = 0.8,
        survival_penalty: float = -0.01,
        death_penalty: float = -5.0,
    ):
        self.max_steps = max_steps
        self.ore_per_battery = ore_per_battery  
        self.batteries_per_heart = batteries_per_heart
        self.enable_combat = enable_combat
        self.clippy_spawn_rate = clippy_spawn_rate
        self.clippy_damage = clippy_damage
        self.heart_reward = heart_reward
        self.ore_reward = ore_reward
        self.battery_reward = battery_reward
        self.survival_penalty = survival_penalty
        self.death_penalty = death_penalty


class TribalEnvConfig:
    """Configuration for the tribal environment."""
    
    def __init__(
        self,
        game: TribalEnvGameConfig | None = None,
        desync_episodes: bool = True,
    ):
        self.game = game or TribalEnvGameConfig()
        self.desync_episodes = desync_episodes


class TribalGennyPlayTool(Tool):
    """
    Clean tribal play tool using pure genny bindings.
    
    This version avoids all the nimpy visualization complexity and uses
    the genny bindings directly, which already have:
    - Environment stepping
    - Observation access  
    - Native Nim visualization
    - External neural network control
    """

    env_config: TribalEnvConfig
    policy_uri: str | None = None

    def invoke(self, args: dict[str, str], overrides: list[str]) -> int | None:
        """Run tribal environment with policy control via genny bindings."""
        print("🎮 Tribal Genny Play Tool")
        print("🎯 Using pure genny bindings for Python-Nim communication")
        
        # Ensure bindings are built
        _ensure_tribal_bindings_built()

        if self.policy_uri:
            if self.policy_uri in ["test_noop", "test_move"]:
                print(f"🧪 Test mode: {self.policy_uri}")
                return self._run_test_neural_network()
            else:
                return self._run_with_neural_network()
        else:
            return self._run_with_builtin_ai()

    def _run_test_neural_network(self) -> int:
        """Run with test neural network using genny bindings."""
        print(f"🧪 Running test neural network: {self.policy_uri}")
        
        # Create test policy
        class TestPolicy:
            def __init__(self, test_mode):
                self.test_mode = test_mode
            
            def forward(self, observations):
                num_agents = 15
                
                if self.test_mode == "test_noop":
                    actions = [[0, 0] for _ in range(num_agents)]
                elif self.test_mode == "test_move":
                    import random
                    actions = [[1, random.randint(0, 3)] for _ in range(num_agents)]
                else:
                    actions = [[0, 0] for _ in range(num_agents)]
                
                return {"actions": actions}
        
        test_policy = TestPolicy(self.policy_uri)
        return self._run_genny_environment(test_policy)

    def _run_with_neural_network(self) -> int:
        """Run with actual neural network policy."""
        try:
            print("🔄 Loading neural network...")
            from metta.rl.checkpoint_manager import CheckpointManager

            policy = CheckpointManager.load_from_uri(self.policy_uri)
            print(f"✅ Neural network loaded: {type(policy).__name__}")

            return self._run_genny_environment(policy)
            
        except Exception as e:
            print(f"❌ Error loading neural network: {e}")
            return 1

    def _run_with_builtin_ai(self) -> int:
        """Run with built-in AI (no neural network)."""
        print("🤖 Running with built-in AI control")
        
        try:
            # Import genny bindings
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent.parent / "tribal" / "bindings" / "generated"))
            import Tribal
            
            # Create environment
            config = self._create_genny_config()
            tribal_env = Tribal.newTribalEnv(config)
            
            # Initialize built-in AI controller
            if not Tribal.initBuiltinAIController():
                print("❌ Failed to initialize built-in AI controller")
                return 1
                
            print("🤖 Built-in AI controller initialized")
            
            # Simple loop - just let the built-in AI run
            tribal_env.resetEnv()
            max_steps = self.env_config.game.max_steps
            
            print(f"🏃 Running built-in AI for {max_steps} steps...")
            for step in range(max_steps):
                # The built-in AI will control agents automatically
                if step % 100 == 0:
                    print(f"  Step {step}")
                    
                if tribal_env.isEpisodeDone():
                    break
            
            print("✅ Built-in AI session completed")
            return 0
            
        except Exception as e:
            print(f"❌ Error running built-in AI: {e}")
            import traceback
            traceback.print_exc()
            return 1

    def _run_genny_environment(self, policy) -> int:
        """Run the environment using pure genny bindings."""
        try:
            print("🎮 Creating tribal environment with pure genny bindings...")
            
            # Import genny bindings
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent.parent / "tribal" / "bindings" / "generated"))
            import Tribal
            
            # Create configuration
            config = self._create_genny_config()
            tribal_env = Tribal.newTribalEnv(config)
            print("✅ Tribal environment created with genny bindings")
            
            # Initialize external neural network controller  
            if not Tribal.initExternalNNController():
                print("❌ Failed to initialize external NN controller")
                return 1
            
            print("🎯 External NN controller initialized - Python will control all agents")
            print("🎮 Starting game loop with native Nim visualization...")
            
            return self._run_policy_loop(tribal_env, policy)
            
        except Exception as e:
            print(f"❌ Error running genny environment: {e}")
            import traceback
            traceback.print_exc()
            return 1

    def _create_genny_config(self):
        """Create genny configuration objects from our config."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "tribal" / "bindings" / "generated"))
        import Tribal
        
        # Create game config
        game_config = Tribal.TribalGameConfig()
        game_config.maxSteps = self.env_config.game.max_steps
        game_config.orePerBattery = self.env_config.game.ore_per_battery
        game_config.batteriesPerHeart = self.env_config.game.batteries_per_heart
        game_config.enableCombat = self.env_config.game.enable_combat
        game_config.clippySpawnRate = self.env_config.game.clippy_spawn_rate
        game_config.clippyDamage = self.env_config.game.clippy_damage
        game_config.heartReward = self.env_config.game.heart_reward
        game_config.batteryReward = self.env_config.game.battery_reward
        game_config.oreReward = self.env_config.game.ore_reward
        game_config.survivalPenalty = self.env_config.game.survival_penalty
        game_config.deathPenalty = self.env_config.game.death_penalty
        
        # Create main config
        config = Tribal.TribalConfig()
        config.game = game_config
        config.desyncEpisodes = self.env_config.desync_episodes
        
        return config

    def _run_policy_loop(self, tribal_env, policy) -> int:
        """Run the main policy control loop."""
        import time
        import numpy as np
        
        try:
            # Reset the environment
            tribal_env.resetEnv()
            step_count = 0
            max_steps = self.env_config.game.max_steps
            
            print(f"✅ Environment reset complete. Starting episode (max {max_steps} steps)")
            print("   The Nim window shows the visualization as we step")
            
            while step_count < max_steps:
                # Get observations using genny bindings
                token_obs = tribal_env.getTokenObservations()
                
                # Convert token observations for policy
                # genny returns flat list: [agent0_tokens..., agent1_tokens...]
                num_agents = 15  # MapAgents constant
                tokens_per_agent = 200  # MaxTokensPerAgent constant  
                token_size = 3  # [coord_byte, layer, value]
                
                # Reshape for policy: (agents, tokens, 3)
                obs_array = np.array(token_obs, dtype=np.float32)
                obs_reshaped = obs_array.reshape(num_agents, tokens_per_agent, token_size)
                
                # Get actions from policy
                if hasattr(policy, 'forward'):
                    # Test policy or custom policy
                    policy_output = policy.forward(obs_reshaped)
                    actions_2d = policy_output["actions"]  # [[action_type, param], ...]
                else:
                    # MettaAgent policy  
                    import torch
                    from metta.rl.env.tensor_dict_env import TensorDict
                    
                    tensor_dict = TensorDict({"observation": torch.tensor(obs_reshaped, dtype=torch.float32)})
                    actions_tensor = policy.forward(tensor_dict)["action"]
                    actions_2d = actions_tensor.numpy().tolist()
                
                # Flatten actions for genny: [a0_type, a0_param, a1_type, a1_param, ...]
                flat_actions = []
                for action in actions_2d:
                    flat_actions.extend([int(action[0]), int(action[1])])
                
                # Step environment using genny bindings
                success = tribal_env.step(flat_actions)
                if not success:
                    print("❌ Environment step failed")
                    break
                
                step_count += 1
                
                # Show progress
                if step_count % 100 == 0:
                    rewards = tribal_env.getRewards()
                    terminated = tribal_env.getTerminated()
                    avg_reward = sum(rewards) / len(rewards) if rewards else 0
                    agents_alive = sum(1 for t in terminated if not t) if terminated else "unknown"
                    print(f"  Step {step_count}: avg_reward={avg_reward:.3f}, agents_alive={agents_alive}")
                    print(f"    Sample actions: agent_0={actions_2d[0]}, agent_1={actions_2d[1]}")
                
                # Check if done
                if tribal_env.isEpisodeDone():
                    print(f"📊 Episode completed after {step_count} steps")
                    break
                
                time.sleep(0.01)  # Small delay for visualization
            
            print(f"🎮 Game completed successfully after {step_count} steps!")
            return 0
            
        except KeyboardInterrupt:
            print("🛑 Stopped by user")
            return 0
        except Exception as e:
            print(f"❌ Policy loop error: {e}")
            import traceback
            traceback.print_exc()
            return 1


def _ensure_tribal_bindings_built():
    """Ensure tribal bindings are built and up to date."""
    tribal_dir = Path(__file__).parent.parent.parent / "tribal"
    bindings_dir = tribal_dir / "bindings" / "generated"
    
    if not bindings_dir.exists():
        print("🔨 Building tribal bindings...")
        import subprocess
        result = subprocess.run(
            ["./build_bindings.sh"],
            cwd=tribal_dir,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"❌ Failed to build bindings: {result.stderr}")
            raise RuntimeError("Binding build failed")
        print("✅ Tribal bindings built successfully")


def make_tribal_environment(
    max_steps: int = 1000,
    ore_per_battery: int = None,
    batteries_per_heart: int = None,
    enable_combat: bool = None,
    clippy_spawn_rate: float = None,
    clippy_damage: int = None,
    heart_reward: float = None,
    ore_reward: float = None,
    battery_reward: float = None,
    survival_penalty: float = None,
    death_penalty: float = None,
    **kwargs,
) -> TribalEnvConfig:
    """Create tribal environment configuration."""
    
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

    game_config = TribalEnvGameConfig(**game_overrides)
    
    return TribalEnvConfig(
        game=game_config,
        desync_episodes=kwargs.get("desync_episodes", True),
    )


def make_tribal_evals():
    """Create default tribal evaluation configurations."""
    return [
        SimulationConfig(
            name="tribal/basic", 
            env=make_tribal_environment(max_steps=2000),
            num_episodes=5,
        )
    ]


def train(
    env: TribalEnvConfig | None = None,
    **overrides
):
    """Train a tribal policy (placeholder - not implemented yet)."""
    raise NotImplementedError("Training not implemented in this clean version yet")


def evaluate(
    policy_uri: str,
    simulations: list[SimulationConfig] | None = None, 
    **overrides
) -> SimTool:
    """Evaluate a tribal policy."""
    # Ensure tribal bindings are built
    _ensure_tribal_bindings_built()

    simulations = simulations or make_tribal_evals()

    return SimTool(
        simulations=simulations,
        policy_uris=[policy_uri],
    )


def play(
    env: TribalEnvConfig | None = None, 
    policy_uri: str | None = None, 
    **overrides
) -> TribalGennyPlayTool:
    """
    Interactive play with the tribal environment using pure genny bindings.
    
    This avoids all nimpy complexity and uses the clean genny API.
    """
    play_env = env or make_tribal_environment()

    return TribalGennyPlayTool(
        env_config=play_env,
        policy_uri=policy_uri,
        **overrides,
    )


def replay(policy_uri: str, **overrides) -> ReplayTool:
    """Replay recorded tribal episodes."""
    _ensure_tribal_bindings_built()

    env = make_tribal_environment()

    return ReplayTool(
        sim=SimulationConfig(name="tribal/replay", env=env),
        policy_uri=policy_uri,
        **overrides,
    )