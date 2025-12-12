import uuid
from typing import Literal

from cogames.cogs_vs_clips.missions import Machina1OpenWorldMission
from metta.doxascope.doxascope_data import DoxascopeLogger
from metta.sim.simulation_config import SimulationConfig
from metta.tools.eval import EvaluateTool
from metta.tools.utils.auto_config import auto_stats_server_uri
from recipes.experiment import (
    arena,  # Original unshaped arena
    cogs_v_clips,  # Cogs vs Clips
)

# Environment types supported for doxascope data collection
EnvironmentType = Literal["arena", "cvc_easy_hearts", "cvc_hello_world", "machina1_open_world"]

# Available environments with descriptions
AVAILABLE_ENVIRONMENTS: dict[str, str] = {
    "arena": "Original arena environment (basic resource collection and combat)",
    "cvc_easy_hearts": "Cogs vs Clips easy_hearts mission (cooperative heart crafting)",
    "cvc_hello_world": "Cogs vs Clips hello_world.easy_hearts (hello world variant)",
    "machina1_open_world": "Machina 1 Open World (v0 leaderboard environment)",
}


def _create_arena_simulation(max_steps: int) -> SimulationConfig:
    """Create a simulation config for the arena environment."""
    all_sims = arena.simulations()
    basic_sim = next((s for s in all_sims if s.name == "basic"), None)

    if not basic_sim:
        raise ValueError("Could not find 'basic' simulation in arena recipe.")

    basic_sim.env.game.max_steps = max_steps
    basic_sim.num_episodes = 1
    basic_sim.doxascope_enabled = True
    # Force bptt=1 to ensure CortexTD runs in rollout mode
    basic_sim.env.game.params = {"bptt": 1}

    return basic_sim


def _create_cvc_simulation(max_steps: int, mission: str = "easy_hearts", num_cogs: int = 4) -> SimulationConfig:
    """Create a simulation config for a Cogs vs Clips mission."""
    env_cfg = cogs_v_clips.make_training_env(
        num_cogs=num_cogs,
        mission=mission,
        variants=["lonely_heart", "heart_chorus", "pack_rat"],  # Common variants
    )

    env_cfg.game.max_steps = max_steps
    # Force bptt=1 to ensure CortexTD runs in rollout mode
    env_cfg.game.params = {"bptt": 1}

    sim_config = SimulationConfig(
        suite="cogs_vs_clips",
        name=f"cvc_{mission.replace('.', '_')}",
        env=env_cfg,
        num_episodes=1,
        doxascope_enabled=True,
    )

    return sim_config


def _create_machina1_simulation(max_steps: int, num_cogs: int = 4, seed: int | None = None) -> SimulationConfig:
    """Create a simulation config for Machina 1 Open World (v0 leaderboard env)."""
    mission = Machina1OpenWorldMission.model_copy(deep=True)
    mission.num_cogs = num_cogs
    if seed and (mb := getattr(mission.site, "map_builder", None)) and hasattr(mb, "seed"):
        mb.seed = seed

    env_cfg = mission.make_env()
    env_cfg.game.max_steps = max_steps
    # Force bptt=1 to ensure CortexTD runs in rollout mode
    env_cfg.game.params = {"bptt": 1}

    sim_config = SimulationConfig(
        suite="v0_leaderboard",
        name="machina1_open_world",
        env=env_cfg,
        num_episodes=1,
        doxascope_enabled=True,
    )

    return sim_config


def simulations(
    max_steps: int | None = None,
    num_simulations: int = 10,
    environment: EnvironmentType = "arena",
) -> list[SimulationConfig]:
    """
    Run a specified number of single-episode simulations for Doxascope data collection.

    Args:
        max_steps: Maximum steps per episode (default: 200 for arena/cvc, 10000 for machina1_open_world)
        num_simulations: Number of simulation runs
        environment: Environment type - "arena", "cvc_easy_hearts", "cvc_hello_world", or "machina1_open_world"

    Note: Uses single-environment configuration as required by doxascope logging.
    """
    # Default max_steps based on environment
    if max_steps is None:
        max_steps = 10000 if environment == "machina1_open_world" else 200

    if environment == "arena":
        base_sim = _create_arena_simulation(max_steps)
    elif environment == "cvc_easy_hearts":
        base_sim = _create_cvc_simulation(max_steps, mission="easy_hearts", num_cogs=4)
    elif environment == "cvc_hello_world":
        base_sim = _create_cvc_simulation(max_steps, mission="hello_world.easy_hearts", num_cogs=4)
    elif environment == "machina1_open_world":
        base_sim = _create_machina1_simulation(max_steps, num_cogs=4)
    else:
        raise ValueError(f"Unknown environment type: {environment}. Available: {list(AVAILABLE_ENVIRONMENTS.keys())}")

    # Create n copies of the simulation config, one for each simulation run
    all_sim_configs = []
    for i in range(num_simulations):
        sim_config = base_sim.model_copy(deep=True)
        sim_config.name = f"{base_sim.name}_{i + 1}"
        all_sim_configs.append(sim_config)
    return all_sim_configs


def evaluate(
    policy_uri: str,
    max_steps: int | None = None,
    num_simulations: int = 10,
    environment: EnvironmentType = "arena",
) -> EvaluateTool:
    """
    Create an evaluation tool for doxascope data collection.

    Args:
        policy_uri: URI of the policy to evaluate
        max_steps: Maximum steps per episode (default: 200 for arena/cvc, 10000 for machina1_open_world)
        num_simulations: Number of simulation runs
        environment: Environment type - "arena", "cvc_easy_hearts", "cvc_hello_world", or "machina1_open_world"
    """
    sims = simulations(max_steps=max_steps, num_simulations=num_simulations, environment=environment)

    # Create and configure the logger here, where we have all the context
    doxascope_logger = DoxascopeLogger(enabled=True, simulation_id=f"eval_{uuid.uuid4().hex[:12]}")
    doxascope_logger.configure(policy_uri=policy_uri, object_type_names=list(sims[0].env.game.objects.keys()))

    return EvaluateTool(
        simulations=sims,
        policy_uris=[policy_uri],
        doxascope_logger=doxascope_logger,
        stats_server_uri=auto_stats_server_uri(),
    )
