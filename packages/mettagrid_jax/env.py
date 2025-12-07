import jax
import jax.numpy as jnp
from jax import lax
import chex
from mettagrid_jax import env_types as types

class MettaGridJAX:
    def __init__(self, params: types.EnvParams = types.EnvParams()):
        self.params = params

    def reset(self, key: chex.PRNGKey) -> types.State:
        key, subkey1, subkey2 = jax.random.split(key, 3)

        # Initialize grid with walls
        # 1. Create fully empty grid
        grid = jnp.zeros((self.params.map_height, self.params.map_width), dtype=jnp.int32)

        # 2. Add random walls (density 0.1)
        wall_mask = jax.random.bernoulli(subkey1, p=0.1, shape=grid.shape)
        grid = jnp.where(wall_mask, types.WALL, types.EMPTY)

        # 3. Ensure borders are walls
        grid = grid.at[0, :].set(types.WALL)
        grid = grid.at[-1, :].set(types.WALL)
        grid = grid.at[:, 0].set(types.WALL)
        grid = grid.at[:, -1].set(types.WALL)

        # 4. Place agents using random sort
        flat_grid = grid.reshape(-1)
        num_cells = flat_grid.shape[0]

        # Assign random priorities to all cells
        random_priorities = jax.random.uniform(subkey2, shape=(num_cells,))

        # Mask out walls (give them very low priority)
        # Using -1.0 for walls so they appear at the bottom
        is_empty = flat_grid == types.EMPTY
        priorities = jnp.where(is_empty, random_priorities, -1.0)

        # Sort indices by priority descending
        # We want the top `num_agents` empty cells
        sorted_indices = jnp.argsort(priorities)[::-1]
        spawn_indices = sorted_indices[:self.params.num_agents]

        spawn_rows = spawn_indices // self.params.map_width
        spawn_cols = spawn_indices % self.params.map_width
        spawn_pos = jnp.stack([spawn_rows, spawn_cols], axis=-1)

        agents = types.AgentState(pos=spawn_pos.astype(jnp.int32))

        return types.State(
            grid=grid,
            agents=agents,
            step_count=jnp.int32(0),
            key=key
        )

    def step(self, state: types.State, actions: chex.Array) -> types.State:
        # actions: (num_agents,)

        # 1. Calculate candidate positions
        deltas = types.DIR_DELTAS[actions] # (N, 2)
        candidate_pos = state.agents.pos + deltas

        # 2. Check Map Bounds
        candidate_pos = jnp.clip(
            candidate_pos,
            jnp.array([0, 0]),
            jnp.array([self.params.map_height - 1, self.params.map_width - 1])
        )

        # 3. Check Walls
        candidate_indices = candidate_pos[:, 0] * self.params.map_width + candidate_pos[:, 1]
        grid_flat = state.grid.reshape(-1)
        cell_types = grid_flat[candidate_indices]

        hit_wall = cell_types == types.WALL
        # Revert if hit wall
        candidate_pos = jnp.where(hit_wall[:, None], state.agents.pos, candidate_pos)

        # 4. Check Agent Collisions (Occupancy)
        occupancy = jnp.zeros_like(grid_flat)
        # Note: We must use candidate_indices updated after wall check?
        # Yes, strictly speaking. But hitting a wall implies reverting to current pos.
        # So we should recompute indices.
        candidate_indices = candidate_pos[:, 0] * self.params.map_width + candidate_pos[:, 1]

        occupancy = occupancy.at[candidate_indices].add(1)

        dest_occupancy = occupancy[candidate_indices]
        collision = dest_occupancy > 1

        final_pos = jnp.where(collision[:, None], state.agents.pos, candidate_pos)

        # Update State
        new_agents = types.AgentState(pos=final_pos)

        new_key, _ = jax.random.split(state.key)

        return state._replace(
            agents=new_agents,
            step_count=state.step_count + 1,
            key=new_key
        )

    def observation(self, state: types.State) -> chex.Array:
        # Local view 5x5 (radius 2)
        view_r = 2
        view_size = 2 * view_r + 1

        # Pad grid
        padded_grid = jnp.pad(state.grid, view_r, constant_values=types.WALL)

        # Construct agent map
        agent_grid = jnp.zeros_like(state.grid)
        agent_indices = state.agents.pos[:, 0] * self.params.map_width + state.agents.pos[:, 1]
        agent_grid = agent_grid.reshape(-1).at[agent_indices].set(1).reshape(state.grid.shape)

        padded_agent_grid = jnp.pad(agent_grid, view_r, constant_values=0)

        # Function to extract window for one agent
        def get_window(pos):
            # pos is (r, c) in original grid
            # in padded grid, top-left of window is (r, c) because padding shifted origin by view_r
            # Wait: Padded origin (0,0) is (-view_r, -view_r).
            # Original (0,0) is at (view_r, view_r) in padded.
            # Window center is at (pos_r + view_r, pos_c + view_r).
            # Window top-left is (pos_r, pos_c).

            top_left_r = pos[0]
            top_left_c = pos[1]

            layer0 = lax.dynamic_slice(padded_grid, (top_left_r, top_left_c), (view_size, view_size))
            layer1 = lax.dynamic_slice(padded_agent_grid, (top_left_r, top_left_c), (view_size, view_size))
            return jnp.stack([layer0, layer1], axis=-1) # (5, 5, 2)

        obs = jax.vmap(get_window)(state.agents.pos)
        return obs
