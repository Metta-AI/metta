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
proc reset*(tribal: TribalEnv, seed: int = -1) =
  ## Reset the environment to initial state
  tribal.env.reset()
  tribal.stepCount = 0
  if seed >= 0:
    # Nim environment uses time-based seeding, but we track this
    discard

proc step*(tribal: TribalEnv, actions: seq[seq[int]]): bool =
  ## Step environment with actions
  ## actions: seq of [action_type, argument] pairs for each agent
  ## Returns true on success, false on error
  
  if actions.len != MapAgents:
    return false
    
  # Convert Python actions to Nim format
  var nimActions: array[MapAgents, array[2, uint8]]
  for i in 0..<MapAgents:
    if i < actions.len and actions[i].len >= 2:
      nimActions[i][0] = actions[i][0].uint8
      nimActions[i][1] = actions[i][1].uint8
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
proc getObservations*(tribal: TribalEnv): seq[seq[seq[seq[int]]]] =
  ## Get current observations as 4D sequence: [agents][layers][height][width]
  result = newSeq[seq[seq[seq[int]]]](MapAgents)
  
  for agentId in 0..<MapAgents:
    result[agentId] = newSeq[seq[seq[int]]](ObservationLayers)
    for layer in 0..<ObservationLayers:
      result[agentId][layer] = newSeq[seq[int]](ObservationHeight)
      for y in 0..<ObservationHeight:
        result[agentId][layer][y] = newSeq[int](ObservationWidth)
        for x in 0..<ObservationWidth:
          result[agentId][layer][y][x] = tribal.env.observations[agentId][layer][x][y].int

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

# Export everything to Python
exportObject(TribalConfig)
exportObject(TribalEnv)

exportProcs:
  newTribalEnv
  defaultConfig  
  getActionSpace

# Export all TribalEnv methods
exportMethods(TribalEnv):
  reset
  step
  getObservations
  getRewards
  getTerminated
  getTruncated
  getCurrentStep
  getMaxSteps
  isEpisodeDone
  getEpisodeStats
  renderText

# Generate the Python binding files
when isMainModule:
  writeFiles("bindings/generated", "Tribal")