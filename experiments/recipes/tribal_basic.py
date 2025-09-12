"""
Tribal Basic Recipe - Nim Environment Integration

Uses genny-generated Nim bindings for high performance.
Agent count (15), map size (100x50), observation shape (19 layers, 11x11) are compile-time constants.
"""

import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from metta.cogworks.curriculum.curriculum import CurriculumConfig
from metta.cogworks.curriculum.task_generator import TaskGenerator
from metta.common.tool import Tool
from metta.cogworks.curriculum.task_generator import TaskGeneratorConfig
from metta.rl.checkpoint_manager import CheckpointManager
from metta.rl.loss.loss_config import LossConfig
from metta.rl.trainer_config import TrainerConfig, EvaluationConfig
from metta.sim.simulation_config import SimulationConfig
from metta.tools.replay import ReplayTool
from metta.tools.sim import SimTool
from metta.tools.train import TrainTool
from metta.tribal import TribalEnvConfig
from metta.tribal.tribal_genny import TribalGameConfig


def _ensure_tribal_bindings_built():
    """Build tribal bindings if they don't exist."""
    metta_root = Path(__file__).parent.parent.parent
    tribal_dir = metta_root / "tribal"
    bindings_dir = tribal_dir / "bindings" / "generated"

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

    bindings_path = str(bindings_dir)
    if bindings_path not in sys.path:
        sys.path.insert(0, bindings_path)


class TribalTaskGeneratorConfig(TaskGeneratorConfig):
    """Tribal-specific task generator config."""

    env: TribalEnvConfig

    def create(self):
        """Create a task generator that returns the tribal environment."""

        class SimpleTribalTaskGenerator(TaskGenerator):
            def __init__(self, config):
                super().__init__(config)
                self._env = config.env

            def _generate_task(self, task_id: int, rng: random.Random):
                return self._env.model_copy(deep=True)

        return SimpleTribalTaskGenerator(self)


class TribalHeadlessPlayTool(Tool):
    """Simple headless tribal environment tool using direct Genny bindings."""

    env_config: TribalEnvConfig
    policy_uri: str | None = None

    def invoke(self, args: dict[str, str], overrides: list[str]) -> int | None:
        """Run tribal environment headlessly with command-line visualization."""
        print("🎮 Tribal Headless Play Tool")
        print("Using direct Genny bindings - no separate process needed!\n")

        if self.policy_uri:
            if self.policy_uri.startswith("test_"):
                print(f"🧪 Test mode: {self.policy_uri}")
                return self._run_test_policy()
            else:
                return self._run_with_neural_network()
        else:
            return self._run_random_policy()

    def _run_test_policy(self) -> int:
        """Run with test policy using direct Genny bindings."""
        from metta.tribal.tribal_genny import make_tribal_env

        # Convert config to dict for make_tribal_env
        config_dict = self.env_config.game.model_dump()
        env = make_tribal_env(**config_dict)

        print(f"✅ Environment created: {env.num_agents} agents")

        obs, info = env.reset()
        max_steps = min(500, self.env_config.game.max_steps)
        print(f"🎮 Running {max_steps} steps with {self.policy_uri}...\n")

        try:
            for step in range(max_steps):
                # Generate actions based on test policy
                actions = []
                for agent_id in range(env.num_agents):
                    if self.policy_uri == "test_noop":
                        actions.append([0, 0])  # NOOP action
                    elif self.policy_uri == "test_move":
                        actions.append([1, random.randint(0, 7)])  # Random movement
                    else:
                        actions.append([0, 0])  # Default NOOP

                actions = np.array(actions, dtype=np.uint8)
                obs, rewards, terminals, truncations, info = env.step(actions)

                # Print status every 20 steps
                if step % 20 == 0:
                    reward_sum = rewards.sum()
                    num_alive = (~(terminals | truncations)).sum()
                    current_step = info.get("current_step", step)
                    print(
                        f"  Step {current_step}: reward_sum={reward_sum:.3f}, agents_alive={num_alive}/{env.num_agents}"
                    )

                    # Show a simple visualization of the first agent's observations
                    if step % 100 == 0:
                        self._print_observations(obs[0], step)

                if info.get("episode_done", False):
                    print(
                        f"\n🏁 Episode ended at step {info.get('current_step', step)}"
                    )
                    break

                # Small delay to make it readable
                time.sleep(0.05)

        except KeyboardInterrupt:
            print("\n⏹️  Interrupted by user")

        print("✅ Test policy run completed!")
        return 0

    def _run_with_neural_network(self) -> int:
        """Run with trained neural network policy."""
        from metta.tribal.tribal_genny import make_tribal_env

        policy = CheckpointManager.load_from_uri(self.policy_uri)
        print(f"✅ Neural network loaded: {type(policy).__name__}")

        # Convert config to dict for make_tribal_env
        config_dict = self.env_config.game.model_dump()
        env = make_tribal_env(**config_dict)

        obs, info = env.reset()
        max_steps = min(200, self.env_config.game.max_steps)
        print(f"🎮 Running {max_steps} steps with neural network...\n")

        try:
            for step in range(max_steps):
                policy_output = policy.forward(obs)
                actions = policy_output["actions"]  # Let it crash if missing

                obs, rewards, terminals, truncations, info = env.step(actions)

                if step % 10 == 0:
                    reward_sum = rewards.sum()
                    num_alive = (~(terminals | truncations)).sum()
                    current_step = info.get("current_step", step)
                    print(
                        f"  Step {current_step}: reward_sum={reward_sum:.3f}, agents_alive={num_alive}/{env.num_agents}"
                    )

                    # Show observations periodically
                    if step % 50 == 0:
                        self._print_observations(obs[0], step)

                if info.get("episode_done", False):
                    print(
                        f"\n🏁 Episode ended at step {info.get('current_step', step)}"
                    )
                    break

                time.sleep(0.02)

        except KeyboardInterrupt:
            print("\n⏹️  Interrupted by user")

        print("✅ Neural network control completed!")
        return 0

    def _run_random_policy(self) -> int:
        """Run with random actions for demonstration."""
        from metta.tribal.tribal_genny import make_tribal_env

        print("🎲 Running with random actions")

        # Convert config to dict for make_tribal_env
        config_dict = self.env_config.game.model_dump()
        env = make_tribal_env(**config_dict)

        obs, info = env.reset()
        max_steps = min(300, self.env_config.game.max_steps)
        print(f"🎮 Running {max_steps} steps with random actions...\n")

        try:
            for step in range(max_steps):
                # Generate random actions
                actions = np.random.randint(
                    0, [6, 8], size=(env.num_agents, 2), dtype=np.uint8
                )
                obs, rewards, terminals, truncations, info = env.step(actions)

                if step % 30 == 0:
                    reward_sum = rewards.sum()
                    num_alive = (~(terminals | truncations)).sum()
                    current_step = info.get("current_step", step)
                    print(
                        f"  Step {current_step}: reward_sum={reward_sum:.3f}, agents_alive={num_alive}/{env.num_agents}"
                    )

                    # Show observations every 60 steps
                    if step % 60 == 0:
                        self._print_observations(obs[0], step)

                if info.get("episode_done", False):
                    print(
                        f"\n🏁 Episode ended at step {info.get('current_step', step)}"
                    )
                    break

                time.sleep(0.1)

        except KeyboardInterrupt:
            print("\n⏹️  Interrupted by user")

        print("✅ Random policy run completed!")
        return 0

    def _print_observations(self, agent_obs, step):
        """Print a simple visualization of agent observations."""
        print(f"\n--- Agent 0 Observations (Step {step}) ---")

        # agent_obs shape: (max_tokens, 3) where 3 = [coord_byte, layer, value]
        active_tokens = []
        for i, token in enumerate(agent_obs):
            coord_byte, layer, value = token
            if coord_byte != 255:  # Valid token (not 0xFF padding)
                x = (coord_byte >> 4) & 0xF
                y = coord_byte & 0xF
                active_tokens.append((x, y, layer, value))

        if active_tokens:
            print(f"Active observations: {len(active_tokens)} tokens")
            # Group by layer and show summary
            layer_counts = {}
            for x, y, layer, value in active_tokens:
                layer_counts[layer] = layer_counts.get(layer, 0) + 1

            layer_names = [
                "Agent",
                "AgentOrient",
                "Ore",
                "Battery",
                "Water",
                "Wheat",
                "Wood",
                "Spear",
                "Hat",
                "Armor",
                "Wall",
                "Mine",
                "MineRes",
                "MineReady",
                "Converter",
                "ConvReady",
                "Altar",
                "AltarHearts",
                "AltarReady",
            ]

            for layer, count in sorted(layer_counts.items()):
                layer_name = (
                    layer_names[layer] if layer < len(layer_names) else f"Layer{layer}"
                )
                print(f"  {layer_name}: {count} observations")
        else:
            print("No active observations")
        print("" + "-" * 45)


def tribal_env_curriculum(tribal_config: TribalEnvConfig) -> CurriculumConfig:
    """Create curriculum configuration from tribal environment config."""
    task_gen_config = TribalTaskGeneratorConfig(env=tribal_config)
    return CurriculumConfig(task_generator=task_gen_config)


def make_tribal_environment(
    max_steps: int = None,
    enable_combat: bool = None,
    ore_per_battery: int = None,
    batteries_per_heart: int = None,
    clippy_spawn_rate: float = None,
    clippy_damage: int = None,
    heart_reward: float = None,
    ore_reward: float = None,
    battery_reward: float = None,
    survival_penalty: float = None,
    death_penalty: float = None,
    **kwargs,
) -> TribalEnvConfig:
    """Create tribal environment configuration with gameplay overrides."""

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

    game_config = TribalGameConfig(**game_overrides)
    config = TribalEnvConfig(
        label="tribal_basic", desync_episodes=True, game=game_config, **kwargs
    )
    return config


def train() -> TrainTool:
    """Train agents on the tribal environment."""
    _ensure_tribal_bindings_built()

    env = make_tribal_environment()
    curriculum = tribal_env_curriculum(env)

    trainer_config = TrainerConfig(
        losses=LossConfig(),
        curriculum=curriculum,
        evaluation=EvaluationConfig(
            simulations=[
                SimulationConfig(name="tribal/basic", env=env),
            ],
            skip_git_check=True,
        ),
    )

    return TrainTool(trainer=trainer_config)


def make_tribal_evals(
    base_env: TribalEnvConfig | None = None,
) -> list[SimulationConfig]:
    """Create evaluation suite for tribal environment."""
    base_env = base_env or make_tribal_environment()

    basic_env = base_env.model_copy()

    combat_env = base_env.model_copy()
    combat_env.game.clippy_spawn_rate = 1.0
    combat_env.game.clippy_damage = 2

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
    """Evaluate a trained policy on the tribal environment."""
    _ensure_tribal_bindings_built()
    simulations = simulations or make_tribal_evals()

    return SimTool(
        simulations=simulations,
        policy_uris=[policy_uri],
    )


def play(
    env: TribalEnvConfig | None = None, policy_uri: str | None = None, **overrides
) -> "TribalHeadlessPlayTool":
    """Interactive play with the tribal environment using headless Genny bindings."""
    play_env = env or make_tribal_environment()
    return TribalHeadlessPlayTool(
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
