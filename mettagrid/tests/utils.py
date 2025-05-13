# tests/utils.py
from typing import Optional

import numpy as np

from mettagrid.mettagrid_c import MettaGrid

# Rebuild the NumPy types using the exposed function
np_actions_type = np.dtype(MettaGrid.get_numpy_type_name("actions"))


def generate_valid_random_actions(
    env,
    num_agents: int,
    force_action_type: Optional[int] = None,
    force_action_arg: Optional[int] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate valid actions for all agents, respecting maximum argument values.

    Args:
        env: MettaGridEnv instance
        num_agents: Number of agents to generate actions for
        force_action_type: If provided, use this action type for all agents
        force_action_arg: If provided, use this action arg (clamped to valid range) for all agents
        seed: Optional random seed for deterministic action generation

    Returns:
        NumPy array of valid actions with shape (num_agents, 2)
    """
    # Set the random seed if provided (for deterministic behavior)
    if seed is not None:
        np.random.seed(seed)

    # Get the action space parameters
    num_actions = env.single_action_space.nvec[0]

    # Get the maximum argument values for each action type
    max_args = env._c_env.max_action_args()

    # Initialize actions array with correct dtype
    actions = np.zeros((num_agents, 2), dtype=np_actions_type)

    for i in range(num_agents):
        # Determine action type
        if force_action_type is None:
            # Random action type if not forced
            act_type = np.random.randint(0, num_actions)
        else:
            # Use forced action type (ensure it's valid)
            act_type = min(force_action_type, num_actions - 1) if num_actions > 0 else 0

        # Get maximum allowed argument for this action type
        max_allowed = max_args[act_type] if act_type < len(max_args) else 0

        # Determine action argument
        if force_action_arg is None:
            # Random valid argument if not forced
            act_arg = np.random.randint(0, max_allowed + 1) if max_allowed >= 0 else 0
        else:
            # Use forced argument (clamped to valid range)
            act_arg = min(force_action_arg, max_allowed)

        # Set the action values
        actions[i, 0] = act_type
        actions[i, 1] = act_arg

    return actions
