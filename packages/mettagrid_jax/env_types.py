import chex
import jax.numpy as jnp
from typing import NamedTuple

# Grid Objects
EMPTY = jnp.int32(0)
WALL = jnp.int32(1)

# Actions
NOOP = jnp.int32(0)
MOVE_UP = jnp.int32(1)
MOVE_DOWN = jnp.int32(2)
MOVE_LEFT = jnp.int32(3)
MOVE_RIGHT = jnp.int32(4)

# Directions mapping (for physics)
# (row, col) delta
DIR_DELTAS = jnp.array([
    [0, 0],   # NOOP
    [-1, 0],  # UP
    [1, 0],   # DOWN
    [0, -1],  # LEFT
    [0, 1],   # RIGHT
], dtype=jnp.int32)

class AgentState(NamedTuple):
    pos: chex.Array  # (num_agents, 2) int32, (row, col)
    # active: chex.Array # (num_agents,) bool - For MVP assuming always active

class State(NamedTuple):
    grid: chex.Array  # (height, width) int32 - Static map (Walls/Empty)
    agents: AgentState
    step_count: chex.Array # scalar int32
    key: chex.PRNGKey

class EnvParams(NamedTuple):
    num_agents: int = 20
    map_width: int = 40
    map_height: int = 40
    max_steps: int = 1000
