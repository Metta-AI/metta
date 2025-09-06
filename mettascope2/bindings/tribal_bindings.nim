## Tribal Environment Python Bindings
## Using genny to create clean Python API for tribal environment

import genny
import std/[json, tables]
import ../src/tribal/environment

# Export types and constants that Python needs to know about
exportConsts:
  MapAgents
  ObservationWidth  
  ObservationHeight
  ObservationLayers
  MapWidth
  MapHeight

# Simple configuration object for Python
type
  TribalConfig* = object
    numAgents*: int
    maxSteps*: int
    mapWidth*: int
    mapHeight*: int
    seed*: int

# Wrapper around Environment for cleaner Python API
type
  TribalEnv* = ref object
    env*: Environment
    config*: TribalConfig
    stepCount*: int

# Constructor
proc newTribalEnv*(config: TribalConfig): TribalEnv =
  ## Create a new tribal environment with given configuration
  result = TribalEnv(
    env: newEnvironment(),  # This calls the existing newEnvironment() 
    config: config,
    stepCount: 0
  )

# Core environment methods
proc resetEnv*(tribal: TribalEnv, seed: int = -1) =
  ## Reset the environment to initial state
  tribal.env.reset()
  tribal.stepCount = 0
  if seed >= 0:
    # Nim environment uses time-based seeding, but we track this
    discard

proc step*(tribal: TribalEnv, actions: seq[int]): bool =
  ## Step environment with actions
  ## actions: flat sequence of [action_type, argument, action_type, argument, ...]
  ## Length should be MapAgents * 2
  ## Returns true on success, false on error
  
  if actions.len != MapAgents * 2:
    return false
    
  # Convert Python actions to Nim format
  var nimActions: array[MapAgents, array[2, uint8]]
  for i in 0..<MapAgents:
    let actionIndex = i * 2
    if actionIndex + 1 < actions.len:
      nimActions[i][0] = actions[actionIndex].uint8
      nimActions[i][1] = actions[actionIndex + 1].uint8
    else:
      nimActions[i][0] = 0  # noop
      nimActions[i][1] = 0
  
  try:
    tribal.env.step(nimActions.addr)
    tribal.stepCount += 1
    return true
  except:
    return false

# Observation access
proc getObservations*(tribal: TribalEnv): seq[int] =
  ## Get current observations as flat sequence: [agents * layers * height * width]
  let totalSize = MapAgents * ObservationLayers * ObservationHeight * ObservationWidth
  result = newSeq[int](totalSize)
  
  var index = 0
  for agentId in 0..<MapAgents:
    for layer in 0..<ObservationLayers:
      for y in 0..<ObservationHeight:
        for x in 0..<ObservationWidth:
          result[index] = tribal.env.observations[agentId][layer][x][y].int
          inc index

# Reward access
proc getRewards*(tribal: TribalEnv): seq[float] =
  ## Get current step rewards for each agent
  result = newSeq[float](MapAgents)
  for i in 0..<min(MapAgents, tribal.env.agents.len):
    result[i] = tribal.env.agents[i].reward
    tribal.env.agents[i].reward = 0.0  # Reset after reading

# Terminal/truncation status
proc getTerminated*(tribal: TribalEnv): seq[bool] =
  ## Get terminated status for each agent
  result = newSeq[bool](MapAgents)
  for i in 0..<MapAgents:
    result[i] = tribal.env.terminated[i] != 0.0

proc getTruncated*(tribal: TribalEnv): seq[bool] =
  ## Get truncated status for each agent  
  result = newSeq[bool](MapAgents)
  for i in 0..<MapAgents:
    result[i] = tribal.env.truncated[i] != 0.0

# Environment info
proc getCurrentStep*(tribal: TribalEnv): int =
  ## Get current step number
  tribal.stepCount

proc getMaxSteps*(tribal: TribalEnv): int =
  ## Get maximum steps per episode
  tribal.config.maxSteps

proc isEpisodeDone*(tribal: TribalEnv): bool =
  ## Check if episode should end
  tribal.stepCount >= tribal.config.maxSteps

# Statistics and debugging
proc getEpisodeStats*(tribal: TribalEnv): string =
  ## Get episode statistics as JSON string
  tribal.env.getEpisodeStats()

proc renderText*(tribal: TribalEnv): string =
  ## Get text rendering of current state
  tribal.env.render()

# Simple configuration helpers
proc defaultConfig*(): TribalConfig =
  ## Get default configuration
  TribalConfig(
    numAgents: MapAgents,
    maxSteps: 1000,
    mapWidth: MapWidth,
    mapHeight: MapHeight,
    seed: 0
  )

# Action space information for Python
proc getActionSpace*(): seq[int] =
  ## Get action space dimensions [num_action_types, max_argument_value]
  ## Tribal has action types 0-5: noop, move, attack, get, swap, put
  @[6, 8]  # 6 action types, 8-directional arguments

# Export sequences first 
exportSeq seq[int]:
  discard

exportSeq seq[float]:
  discard

exportSeq seq[bool]:
  discard

# Export everything to Python
exportObject TribalConfig:
  discard

exportRefObject TribalEnv:
  constructor:
    newTribalEnv(TribalConfig)
  procs:
    resetEnv(TribalEnv, int)
    step(TribalEnv, seq[int])
    getObservations(TribalEnv)
    getRewards(TribalEnv)
    getTerminated(TribalEnv)
    getTruncated(TribalEnv)
    getCurrentStep(TribalEnv)
    getMaxSteps(TribalEnv)
    isEpisodeDone(TribalEnv)
    getEpisodeStats(TribalEnv)
    renderText(TribalEnv)

exportProcs:
  defaultConfig
  getActionSpace

# Generate the Python binding files and include implementation
writeFiles("bindings/generated", "Tribal")
include generated/internal