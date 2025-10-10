from typing import Optional, Sequence

from metta.sim.simulation_config import SimulationConfig
from metta.tools.sim import SimTool

from experiments.recipes import arena


def evaluate(
    policy_uri: str,
    max_steps: int = 100,
    num_simulations: int = 10,
    simulations: Optional[Sequence[SimulationConfig]] = None,
) -> SimTool:
    """
    Run a specified number of single-episode simulations for Doxascope data collection.

    Note: Uses single-environment configuration as required by doxascope logging.
    """
    # Use the arena recipe to get the base "basic" simulation config
    all_sims = arena.make_evals()
    basic_sim = next((s for s in all_sims if s.name == "basic"), None)

    if not basic_sim:
        raise ValueError("Could not find 'basic' simulation in arena recipe.")

    # Override the parameters for Doxascope data collection
    basic_sim.env.game.max_steps = max_steps
    basic_sim.num_episodes = 1
    basic_sim.doxascope_enabled = True
    # Ensure all agents are policy agents for data collection
    basic_sim.npc_policy_uri = None
    basic_sim.policy_agents_pct = 1.0

    # Create n copies of the simulation config, one for each simulation run
    all_sim_configs = []
    for i in range(num_simulations):
        sim_config = basic_sim.copy(deep=True)
        sim_config.name = f"{basic_sim.name}_{i + 1}"
        all_sim_configs.append(sim_config)

    return SimTool(
        simulations=all_sim_configs,
        policy_uris=[policy_uri],
    )
