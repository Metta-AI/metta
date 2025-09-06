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

# Configuration objects matching Python TribalEnvConfig structure
# NOTE: Structural parameters (numAgents, obsWidth, etc.) are kept as compile-time constants
type
  TribalGameConfig* = object
    # Core game parameters
    maxSteps*: int
    
    # Resource configuration
    orePerBattery*: int
    batteriesPerHeart*: int
    
    # Combat configuration
    enableCombat*: bool
    clippySpawnRate*: float
    clippyDamage*: int
    
    # Reward configuration
    heartReward*: float
    oreReward*: float
    batteryReward*: float
    survivalPenalty*: float
    deathPenalty*: float
  
  TribalConfig* = object
    game*: TribalGameConfig
    desyncEpisodes*: bool

# Wrapper around Environment for cleaner Python API
type
  TribalEnv* = ref object
    env*: Environment
    config*: TribalConfig
    stepCount*: int

# Helper procedure to create default config
proc defaultTribalConfig*(): TribalConfig =
  ## Create default tribal configuration
  TribalConfig(
    game: TribalGameConfig(
      maxSteps: 2000,
      orePerBattery: 3,
      batteriesPerHeart: 2,
      enableCombat: true,
      clippySpawnRate: 0.05,
      clippyDamage: 1,
      heartReward: 1.0,
      oreReward: 0.1,  # Exact arena_basic_easy_shaped value
      batteryReward: 0.8,  # Exact arena_basic_easy_shaped value
      survivalPenalty: -0.01,
      deathPenalty: -5.0
    ),
    desyncEpisodes: true
  )

# Constructors
proc newTribalEnv*(config: TribalConfig): TribalEnv =
  ## Create a new tribal environment with full configuration
  try:
    # Convert TribalConfig to EnvironmentConfig (only configurable parameters)
    var envConfig = EnvironmentConfig(
      maxSteps: config.game.maxSteps,
      orePerBattery: config.game.orePerBattery,
      batteriesPerHeart: config.game.batteriesPerHeart,
      enableCombat: config.game.enableCombat,
      clippySpawnRate: config.game.clippySpawnRate,
      clippyDamage: config.game.clippyDamage,
      heartReward: config.game.heartReward,
      oreReward: config.game.oreReward,
      batteryReward: config.game.batteryReward,
      woodReward: 0.002,  # Default values for rewards not exposed in Python yet
      waterReward: 0.001,
      wheatReward: 0.001,
      spearReward: 0.01,
      armorReward: 0.015,
      foodReward: 0.012,
      clothReward: 0.012,
      clippyKillReward: 0.1,
      survivalPenalty: config.game.survivalPenalty,
      deathPenalty: config.game.deathPenalty
    )
    
    result = TribalEnv(
      env: newEnvironment(envConfig),
      config: config,
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
  tribal.stepCount >= tribal.config.game.maxSteps

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
exportObject TribalGameConfig:
  discard

exportObject TribalConfig:
  discard

# Export ref objects
exportRefObject TribalEnv:
  constructor:
    newTribalEnv(TribalConfig)
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
  defaultTribalConfig

# Generate the Python binding files and include implementation (must be at the end)
writeFiles("bindings/generated", "Tribal")
include generated/internal