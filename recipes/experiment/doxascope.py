import uuid

from metta.doxascope.doxascope_data import DoxascopeLogger
from metta.sim.simulation_config import SimulationConfig
from metta.tools.eval import EvaluateTool
from recipes.experiment import arena  # Original unshaped arena
# from recipes.prod import arena_basic_easy_shaped # ABES option


def simulations(
    max_steps: int = 100,
    num_simulations: int = 10,
) -> list[SimulationConfig]:
    """
    Run a specified number of single-episode simulations for Doxascope data collection.

    Note: Uses single-environment configuration as required by doxascope logging.
    """
    all_sims = arena.simulations()  # Original unshaped arena
    # all_sims = arena_basic_easy_shaped.simulations() # ABES option
    basic_sim = next((s for s in all_sims if s.name == "basic"), None)

    if not basic_sim:
        raise ValueError("Could not find 'basic' simulation in arena recipe.")

    # Override the parameters for Doxascope data collection
    basic_sim.env.game.max_steps = max_steps
    basic_sim.num_episodes = 1
    basic_sim.doxascope_enabled = True
    # Force bptt=1 to ensure CortexTD runs in rollout mode
    basic_sim.env.game.params = {"bptt": 1}

    # Create n copies of the simulation config, one for each simulation run
    all_sim_configs = []
    for i in range(num_simulations):
        sim_config = basic_sim.model_copy(deep=True)
        sim_config.name = f"{basic_sim.name}_{i + 1}"
        all_sim_configs.append(sim_config)
    return all_sim_configs


def evaluate(
    policy_uri: str,
    max_steps: int = 100,
    num_simulations: int = 10,
) -> EvaluateTool:
    sims = simulations(max_steps=max_steps, num_simulations=num_simulations)

    # Create and configure the logger here, where we have all the context
    doxascope_logger = DoxascopeLogger(enabled=True, simulation_id=f"eval_{uuid.uuid4().hex[:12]}")
    doxascope_logger.configure(policy_uri=policy_uri, object_type_names=list(sims[0].env.game.objects.keys()))

    return EvaluateTool(
        simulations=sims,
        policy_uris=[policy_uri],
        doxascope_logger=doxascope_logger,
        stats_server_uri=None,
        enable_replays=False,
    )
