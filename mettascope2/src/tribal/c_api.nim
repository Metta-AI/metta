## C API for Tribal Environment
## Provides C bindings that match the mettagrid interface pattern
## for integration with Python training pipeline

import environment
import std/[json, strformat]

# C-compatible types
type
  TribalGrid* = ptr Environment
  CTribalConfig* = object
    num_agents*: cint
    max_steps*: cint
    map_width*: cint
    map_height*: cint
    observation_width*: cint
    observation_height*: cint
    observation_layers*: cint

# Global buffers that Python will allocate and pass to us
var
  g_observations_buffer*: ptr UncheckedArray[uint8]
  g_rewards_buffer*: ptr UncheckedArray[cfloat]
  g_terminals_buffer*: ptr UncheckedArray[uint8]  # bool as uint8
  g_truncations_buffer*: ptr UncheckedArray[uint8]  # bool as uint8

# Convert Nim Environment to C pointer
proc toTribalGrid(env: Environment): TribalGrid =
  cast[TribalGrid](env)

proc fromTribalGrid(grid: TribalGrid): Environment =
  cast[Environment](grid)

# C API Functions - must match mettagrid interface exactly
{.exportc, dynlib.}
proc tribal_grid_create(config: ptr CTribalConfig): TribalGrid =
  ## Create new tribal environment instance
  let env = newEnvironment()
  result = env.toTribalGrid()

{.exportc, dynlib.}
proc tribal_grid_destroy(grid: TribalGrid) =
  ## Clean up tribal environment instance
  if grid != nil:
    # Nim's GC will handle cleanup
    discard

{.exportc, dynlib.}
proc tribal_grid_set_buffers(
  grid: TribalGrid,
  observations: ptr UncheckedArray[uint8],
  terminals: ptr UncheckedArray[uint8],
  truncations: ptr UncheckedArray[uint8], 
  rewards: ptr UncheckedArray[cfloat]
) =
  ## Set shared memory buffers - Python allocates, we write to them
  g_observations_buffer = observations
  g_terminals_buffer = terminals
  g_truncations_buffer = truncations
  g_rewards_buffer = rewards

{.exportc, dynlib.}
proc tribal_grid_reset(grid: TribalGrid, seed: cint): cint =
  ## Reset environment and return observations
  ## Returns 1 on success, 0 on error
  try:
    let env = grid.fromTribalGrid()
    env.reset()
    
    # Copy observations to shared buffer
    if g_observations_buffer != nil:
      let obs_size = MapAgents * ObservationLayers * ObservationWidth * ObservationHeight
      copyMem(g_observations_buffer, env.observations[0][0][0].addr, obs_size * sizeof(uint8))
    
    # Clear other buffers
    if g_terminals_buffer != nil:
      for i in 0..<MapAgents:
        g_terminals_buffer[i] = 0
    
    if g_truncations_buffer != nil:
      for i in 0..<MapAgents:
        g_truncations_buffer[i] = 0
    
    if g_rewards_buffer != nil:
      for i in 0..<MapAgents:
        g_rewards_buffer[i] = 0.0
    
    return 1
  except:
    return 0

{.exportc, dynlib.}
proc tribal_grid_step(grid: TribalGrid, actions: ptr UncheckedArray[uint8]): cint =
  ## Step environment with actions and update buffers
  ## Returns 1 on success, 0 on error
  try:
    let env = grid.fromTribalGrid()
    
    # Convert action buffer to Nim format
    var nim_actions: array[MapAgents, array[2, uint8]]
    for i in 0..<MapAgents:
      nim_actions[i][0] = actions[i * 2]     # action_type
      nim_actions[i][1] = actions[i * 2 + 1] # argument
    
    # Step the environment
    env.step(nim_actions.addr)
    
    # Copy results to shared buffers
    if g_observations_buffer != nil:
      let obs_size = MapAgents * ObservationLayers * ObservationWidth * ObservationHeight
      copyMem(g_observations_buffer, env.observations[0][0][0].addr, obs_size * sizeof(uint8))
    
    if g_rewards_buffer != nil:
      for i in 0..<MapAgents:
        if i < env.agents.len:
          g_rewards_buffer[i] = env.agents[i].reward
          env.agents[i].reward = 0.0  # Reset reward after reading
        else:
          g_rewards_buffer[i] = 0.0
    
    if g_terminals_buffer != nil:
      for i in 0..<MapAgents:
        g_terminals_buffer[i] = if env.terminated[i] != 0.0: 1 else: 0
    
    if g_truncations_buffer != nil:
      for i in 0..<MapAgents:
        g_truncations_buffer[i] = if env.truncated[i] != 0.0: 1 else: 0
    
    return 1
  except:
    return 0

{.exportc, dynlib.}
proc tribal_grid_current_step(grid: TribalGrid): cint =
  ## Get current step count
  if grid != nil:
    let env = grid.fromTribalGrid()
    return env.currentStep.cint
  return 0

{.exportc, dynlib.}
proc tribal_grid_max_steps(grid: TribalGrid): cint =
  ## Get maximum steps (for compatibility)
  return 1000  # Default max steps

{.exportc, dynlib.}
proc tribal_grid_get_episode_stats(grid: TribalGrid): cstring =
  ## Get episode statistics as JSON string
  if grid != nil:
    let env = grid.fromTribalGrid()
    let stats = env.getEpisodeStats()
    return stats.cstring
  return "{}".cstring

# Export data type information for Python
{.exportc, dynlib.}
proc tribal_grid_get_dtype_info(): cstring =
  ## Return data type information for Python numpy arrays
  let info = %* {
    "observations": "uint8",
    "terminals": "bool", 
    "truncations": "bool",
    "rewards": "float32",
    "actions": "int32"
  }
  return ($info).cstring

# Export environment dimensions
{.exportc, dynlib.}
proc tribal_grid_get_dimensions(): cstring =
  ## Return environment dimensions
  let dims = %* {
    "num_agents": MapAgents,
    "observation_width": ObservationWidth,
    "observation_height": ObservationHeight,
    "observation_layers": ObservationLayers,
    "map_width": MapWidth,
    "map_height": MapHeight
  }
  return ($dims).cstring