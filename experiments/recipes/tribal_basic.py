"""
Tribal Basic Recipe - Nim Environment Integration

Uses genny-generated Nim bindings for high performance.
Agent count (15), map size (100x50), observation shape (19 layers, 11x11) are compile-time constants.
"""

import subprocess
import sys
import time
from pathlib import Path

# Add metta root to Python path so 'tribal' module can be found
_metta_root = Path(__file__).parent.parent.parent
if str(_metta_root) not in sys.path:
    sys.path.insert(0, str(_metta_root))

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

    import sys

    bindings_path = str(bindings_dir)
    if bindings_path not in sys.path:
        sys.path.insert(0, bindings_path)


class TribalTaskGeneratorConfig(TaskGeneratorConfig):
    """Tribal-specific task generator config."""

    env: TribalEnvConfig

    def create(self):
        """Create a task generator that returns the tribal environment."""
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
    """Process-separated tribal environment tool using file-based IPC."""

    env_config: TribalEnvConfig
    policy_uri: str | None = None

    def invoke(self, args: dict[str, str], overrides: list[str]) -> int | None:
        """Run tribal environment with process separation."""
        print("🎮 Tribal Process-Separated Play Tool")

        project_root = Path(__file__).parent.parent.parent
        tribal_dir = project_root / "tribal"

        try:
            from tribal.src.tribal_process_controller import TribalProcessController
        except ImportError as e:
            print(f"❌ Failed to import process controller: {e}")
            return 1

        if self.policy_uri:
            if self.policy_uri in ["test_noop", "test_move"]:
                print(f"🧪 Test mode: {self.policy_uri}")
                return self._run_test_policy(TribalProcessController, tribal_dir)
            else:
                return self._run_with_neural_network(
                    TribalProcessController, tribal_dir
                )
        else:
            return self._run_builtin_ai(TribalProcessController, tribal_dir)

    def _run_test_policy(self, ControllerClass, tribal_dir: Path) -> int:
        """Run with test policy."""
        try:
            with ControllerClass(tribal_dir) as controller:
                if not controller.start_nim_process():
                    print("❌ Failed to start Nim viewer process")
                    return 1

                controller.activate_communication()
                obs, info = controller.reset()

                max_steps = min(500, self.env_config.game.max_steps)
                print(f"🎮 Running {max_steps} steps with {self.policy_uri}...")

                for step in range(max_steps):
                    actions = []
                    for agent in range(controller.num_agents):
                        if self.policy_uri == "test_noop":
                            actions.append([0, 0])
                        elif self.policy_uri == "test_move":
                            import random

                            actions.append([1, random.randint(0, 3)])
                        else:
                            actions.append([0, 0])

                    obs, rewards, terminals, truncations, info = controller.step(
                        actions
                    )

                    if step % 20 == 0:
                        reward_sum = rewards.sum()
                        num_alive = (~(terminals | truncations)).sum()
                        print(
                            f"  Step {step}: reward_sum={reward_sum:.3f}, agents_alive={num_alive}"
                        )

                    if info.get("episode_done", False):
                        print(f"🏁 Episode ended at step {step}")
                        break

                    time.sleep(0.1)

                print("✅ Process-separated communication working!")
                time.sleep(5)
                return 0

        except Exception as e:
            print(f"❌ Error in test policy run: {e}")
            import traceback

            traceback.print_exc()
            return 1

    def _run_with_neural_network(self, ControllerClass, tribal_dir: Path) -> int:
        """Run with trained neural network policy."""
        try:
            from metta.rl.checkpoint_manager import CheckpointManager

            policy = CheckpointManager.load_from_uri(self.policy_uri)
            print(f"✅ Neural network loaded: {type(policy).__name__}")

            with ControllerClass(tribal_dir) as controller:
                if not controller.start_nim_process():
                    print("❌ Failed to start Nim viewer process")
                    return 1

                controller.activate_communication()
                obs, info = controller.reset()

                max_steps = min(200, self.env_config.game.max_steps)
                print(f"🎮 Running {max_steps} steps with neural network...")

                for step in range(max_steps):
                    policy_output = policy.forward(obs)
                    actions = policy_output.get("actions")

                    if actions is None:
                        print("⚠️  Policy didn't return actions, using noop")
                        actions = [[0, 0] for _ in range(controller.num_agents)]

                    obs, rewards, terminals, truncations, info = controller.step(
                        actions
                    )

                    if step % 50 == 0:
                        reward_sum = rewards.sum()
                        num_alive = (~(terminals | truncations)).sum()
                        print(
                            f"  Step {step}: reward_sum={reward_sum:.3f}, agents_alive={num_alive}"
                        )

                    if info.get("episode_done", False):
                        print(f"🏁 Episode ended at step {step}")
                        break

                print("✅ Neural network control working!")
                time.sleep(3)
                return 0

        except Exception as e:
            print(f"❌ Error with neural network: {e}")
            import traceback

            traceback.print_exc()
            return 1

    def _run_builtin_ai(self, ControllerClass, tribal_dir: Path) -> int:
        """Run with built-in AI."""
        print("🤖 Running with built-in AI")

        try:
            with ControllerClass(tribal_dir) as controller:
                if not controller.start_nim_process():
                    print("❌ Failed to start Nim viewer process")
                    return 1

                controller.activate_communication()
                print("Press Ctrl+C to stop")

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
    from tribal.src.tribal_genny import TribalGameConfig

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
) -> "TribalProcessPlayTool":
    """Interactive play with the tribal environment using process separation."""
    play_env = env or make_tribal_environment()
    return TribalProcessPlayTool(
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
