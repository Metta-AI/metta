## Tribal Environment Python Bindings
## Using genny to create clean Python API for tribal environment

import genny
import ../src/tribal/environment

# Global error handling (following pixie pattern)
var lastError: ref Exception

proc takeError(): string =
  if lastError != nil:
    result = lastError.msg
    lastError = nil
  else:
    result = ""

proc checkError(): bool =
  result = lastError != nil

# Export constants that Python needs to know about
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
    maxSteps*: int
    seed*: int

# Wrapper around Environment for cleaner Python API
type
  TribalEnv* = ref object
    env*: Environment
    config*: TribalConfig
    stepCount*: int

# Constructor
proc newTribalEnv*(maxSteps: int): TribalEnv =
  ## Create a new tribal environment
  try:
    result = TribalEnv(
      env: newEnvironment(),
      config: TribalConfig(maxSteps: maxSteps, seed: 0),
      stepCount: 0
    )
  except:
    lastError = getCurrentException()

# Core environment methods
proc resetEnv*(tribal: TribalEnv) =
  ## Reset the environment to initial state
  try:
    tribal.env.reset()
    tribal.stepCount = 0
  except:
    lastError = getCurrentException()

proc step*(tribal: TribalEnv, actions: seq[int]): bool =
  ## Step environment with actions
  ## actions: flat sequence of [action_type, argument] pairs
  ## Length should be MapAgents * 2
  try:
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
    
    tribal.env.step(nimActions.addr)
    tribal.stepCount += 1
    return true
  except:
    lastError = getCurrentException()
    return false

# Observation access
proc getObservations*(tribal: TribalEnv): seq[int] =
  ## Get current observations as flat sequence
  try:
    let totalSize = MapAgents * ObservationLayers * ObservationHeight * ObservationWidth
    result = newSeq[int](totalSize)
    
    var index = 0
    for agentId in 0..<MapAgents:
      for layer in 0..<ObservationLayers:
        for y in 0..<ObservationHeight:
          for x in 0..<ObservationWidth:
            result[index] = tribal.env.observations[agentId][layer][x][y].int
            inc index
  except:
    lastError = getCurrentException()

# Reward access
proc getRewards*(tribal: TribalEnv): seq[float] =
  ## Get current step rewards for each agent
  try:
    result = newSeq[float](MapAgents)
    for i in 0..<min(MapAgents, tribal.env.agents.len):
      result[i] = tribal.env.agents[i].reward
      tribal.env.agents[i].reward = 0.0  # Reset after reading
  except:
    lastError = getCurrentException()

# Terminal/truncation status
proc getTerminated*(tribal: TribalEnv): seq[bool] =
  ## Get terminated status for each agent
  try:
    result = newSeq[bool](MapAgents)
    for i in 0..<MapAgents:
      result[i] = tribal.env.terminated[i] != 0.0
  except:
    lastError = getCurrentException()

proc getTruncated*(tribal: TribalEnv): seq[bool] =
  ## Get truncated status for each agent  
  try:
    result = newSeq[bool](MapAgents)
    for i in 0..<MapAgents:
      result[i] = tribal.env.truncated[i] != 0.0
  except:
    lastError = getCurrentException()

# Environment info
proc getCurrentStep*(tribal: TribalEnv): int =
  ## Get current step number
  tribal.stepCount

proc isEpisodeDone*(tribal: TribalEnv): bool =
  ## Check if episode should end
  tribal.stepCount >= tribal.config.maxSteps

# Statistics and debugging
proc renderText*(tribal: TribalEnv): string =
  ## Get text rendering of current state
  try:
    result = tribal.env.render()
  except:
    lastError = getCurrentException()
    result = ""

# Helper procedures
proc defaultMaxSteps*(): int =
  ## Get default max steps value
  1000

# Export sequences first (following pixie pattern)
exportSeq seq[int]:
  discard

exportSeq seq[float]:
  discard

exportSeq seq[bool]:
  discard

# Export error handling procedures
exportProcs:
  checkError
  takeError

# Export simple objects
exportObject TribalConfig:
  discard

# Export ref objects
exportRefObject TribalEnv:
  constructor:
    newTribalEnv(int)
  procs:
    resetEnv(TribalEnv)
    step(TribalEnv, seq[int])
    getObservations(TribalEnv)
    getRewards(TribalEnv)
    getTerminated(TribalEnv)
    getTruncated(TribalEnv)
    getCurrentStep(TribalEnv)
    isEpisodeDone(TribalEnv)
    renderText(TribalEnv)

# Export standalone procedures
exportProcs:
  defaultMaxSteps

# Generate the Python binding files and include implementation (must be at the end)
writeFiles("bindings/generated", "Tribal")
include generated/internal