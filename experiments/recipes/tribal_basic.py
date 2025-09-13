"""
Tribal Basic Recipe

Tribal environment with easy converter chain (1 battery per heart vs default 2).
Tribal defaults already provide good shaped rewards matching arena_basic_easy_shaped.

Uses genny-generated Nim bindings for high performance.
Agent count (15), map size (100x50) are compile-time constants.
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
            raise RuntimeError(f"Tribal bindings build failed: {result.stderr}")

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
        print("Tribal Headless Play Tool")

        if self.policy_uri:
            if self.policy_uri.startswith("test_"):
                print(f"Test mode: {self.policy_uri}")
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

        # Environment created with {env.num_agents} agents

        obs, info = env.reset()
        max_steps = min(500, self.env_config.game.max_steps)
        print(f"Running {max_steps} steps with {self.policy_uri}...\n")

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

                # Show full environment grid every step for real-time visualization
                self._print_full_environment_grid(env, step)

                if info.get("episode_done", False):
                    print(f"\nEpisode ended at step {info.get('current_step', step)}")
                    break

                # Small delay to make it readable
                time.sleep(0.05)

        except KeyboardInterrupt:
            print("\nInterrupted by user")

        print("✅ Test policy run completed!")
        return 0

    def _run_with_neural_network(self) -> int:
        """Run with trained neural network policy."""
        from metta.tribal.tribal_genny import make_tribal_env

        policy = CheckpointManager.load_from_uri(self.policy_uri)
        # Neural network loaded: {type(policy).__name__}

        # Convert config to dict for make_tribal_env
        config_dict = self.env_config.game.model_dump()
        env = make_tribal_env(**config_dict)

        obs, info = env.reset()
        max_steps = min(200, self.env_config.game.max_steps)
        print(f"Running {max_steps} steps with neural network...\n")

        try:
            for step in range(max_steps):
                policy_output = policy.forward(obs)
                actions = policy_output["actions"]  # Let it crash if missing

                obs, rewards, terminals, truncations, info = env.step(actions)

                # Show full environment grid every step for real-time visualization
                self._print_full_environment_grid(env, step)

                if info.get("episode_done", False):
                    print(f"\nEpisode ended at step {info.get('current_step', step)}")
                    break

                time.sleep(0.02)

        except KeyboardInterrupt:
            print("\nInterrupted by user")

        print("Neural network control completed!")
        return 0

    def _run_random_policy(self) -> int:
        """Run with random actions for demonstration."""
        from metta.tribal.tribal_genny import make_tribal_env

        # Running with random actions

        # Convert config to dict for make_tribal_env
        config_dict = self.env_config.game.model_dump()
        env = make_tribal_env(**config_dict)

        obs, info = env.reset()
        max_steps = min(300, self.env_config.game.max_steps)
        print(f"Running {max_steps} steps with random actions...\n")

        try:
            for step in range(max_steps):
                # Generate random actions
                actions = np.random.randint(
                    0, [6, 8], size=(env.num_agents, 2), dtype=np.uint8
                )
                obs, rewards, terminals, truncations, info = env.step(actions)

                # Show full environment grid every step for real-time visualization
                self._print_full_environment_grid(env, step)

                if info.get("episode_done", False):
                    print(f"\nEpisode ended at step {info.get('current_step', step)}")
                    break

                time.sleep(0.1)

        except KeyboardInterrupt:
            print("\nInterrupted by user")

        print("Random policy run completed!")
        return 0

    # Removed verbose _print_observations method - using render_text() instead

    def _print_full_environment_grid(self, env, step):
        """Print the complete environment grid using the render_text method."""
        print("\033[2J\033[H", end="")  # Clear screen

        try:
            full_grid_text = env._nim_env.render_text()
            if full_grid_text and full_grid_text.strip():
                print(f"Step {step} - Full Environment Grid:")
                print(full_grid_text)
            else:
                print(f"Step {step} - No environment text available")
        except Exception as e:
            print(f"Step {step} - Error getting environment grid: {e}")


def tribal_env_curriculum(tribal_config: TribalEnvConfig) -> CurriculumConfig:
    """Create curriculum configuration from tribal environment config."""
    task_gen_config = TribalTaskGeneratorConfig(env=tribal_config)
    return CurriculumConfig(task_generator=task_gen_config)


def make_tribal_environment(**kwargs) -> TribalEnvConfig:
    """Create tribal environment with easy converter chain (1 battery per heart).

    Tribal defaults already provide arena_basic_easy_shaped reward structure:
    - heart_reward: 1.0, ore_reward: 0.1, battery_reward: 0.8
    - survival_penalty: -0.01, death_penalty: -5.0
    """
    game_config = TribalGameConfig(batteries_per_heart=1, **kwargs)
    return TribalEnvConfig(label="tribal_basic", desync_episodes=True, game=game_config)


def train() -> TrainTool:
    """Train agents on the tribal environment."""
    _ensure_tribal_bindings_built()

    env = make_tribal_environment()
    curriculum = tribal_env_curriculum(env)

    trainer_config = TrainerConfig(
        losses=LossConfig(),
        curriculum=curriculum,
        evaluation=EvaluationConfig(
            simulations=make_tribal_evals(env),
            skip_git_check=True,
        ),
    )

    return TrainTool(trainer=trainer_config)


def make_tribal_evals(
    base_env: TribalEnvConfig | None = None,
) -> list[SimulationConfig]:
    """Create evaluation suite using the same environment as training."""
    base_env = base_env or make_tribal_environment()

    return [
        SimulationConfig(name="tribal/basic", env=base_env, num_episodes=10),
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
