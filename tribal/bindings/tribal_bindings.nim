## Tribal Environment Python Bindings
## Using genny to create clean Python API for tribal environment

import genny
import ../src/tribal/environment

# Global error handling
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

# Additional constants for Python interface
const MaxTokensPerAgent* = 200
const NumActionTypes* = 6

exportConsts:
  MaxTokensPerAgent
  NumActionTypes

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
  ## Create default tribal configuration by referencing environment defaults
  let envDefaults = defaultEnvironmentConfig()
  TribalConfig(
    game: TribalGameConfig(
      maxSteps: envDefaults.maxSteps,
      orePerBattery: envDefaults.orePerBattery,
      batteriesPerHeart: envDefaults.batteriesPerHeart,
      enableCombat: envDefaults.enableCombat,
      clippySpawnRate: envDefaults.clippySpawnRate,
      clippyDamage: envDefaults.clippyDamage,
      heartReward: envDefaults.heartReward,
      oreReward: envDefaults.oreReward,
      batteryReward: envDefaults.batteryReward,
      survivalPenalty: envDefaults.survivalPenalty,
      deathPenalty: envDefaults.deathPenalty
    ),
    desyncEpisodes: true
  )

# Constructors
proc newTribalEnv*(config: TribalConfig): TribalEnv =
  ## Create a new tribal environment with full configuration
  try:
    # Start with default config and override with configurable parameters
    var envConfig = defaultEnvironmentConfig()
    envConfig.maxSteps = config.game.maxSteps
    envConfig.orePerBattery = config.game.orePerBattery
    envConfig.batteriesPerHeart = config.game.batteriesPerHeart
    envConfig.enableCombat = config.game.enableCombat
    envConfig.clippySpawnRate = config.game.clippySpawnRate
    envConfig.clippyDamage = config.game.clippyDamage
    envConfig.heartReward = config.game.heartReward
    envConfig.oreReward = config.game.oreReward
    envConfig.batteryReward = config.game.batteryReward
    envConfig.survivalPenalty = config.game.survivalPenalty
    envConfig.deathPenalty = config.game.deathPenalty
    
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
  ## Get current observations as flat sequence (deprecated, use getTokenObservations)
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

proc getTokenObservations*(tribal: TribalEnv): seq[int] =
  ## Get current observations as token sequence compatible with MettaGrid format
  ## Returns flattened array of [agent0_tokens..., agent1_tokens..., ...]
  ## where each agent has MaxTokensPerAgent tokens of 3 values: [coord_byte, layer, value]
  try:
    const TokenSize = 3  # [coord_byte, layer, value]
    
    result = newSeq[int](MapAgents * MaxTokensPerAgent * TokenSize)
    
    for agentId in 0..<MapAgents:
      var tokenCount = 0
      let baseIndex = agentId * MaxTokensPerAgent * TokenSize
      
      # Convert observations to tokens for this agent
      for layer in 0..<ObservationLayers:
        for y in 0..<ObservationHeight:
          for x in 0..<ObservationWidth:
            let value = tribal.env.observations[agentId][layer][x][y]
            if value > 0 and tokenCount < MaxTokensPerAgent:
              # Pack coordinates into single byte (4 bits each, max 15)
              let coordByte = (x shl 4) or y
              let tokenIndex = baseIndex + tokenCount * TokenSize
              
              result[tokenIndex] = coordByte
              result[tokenIndex + 1] = layer
              result[tokenIndex + 2] = value.int
              
              inc tokenCount
      
      # Fill remaining tokens with 0xFF (invalid marker)
      for i in tokenCount..<MaxTokensPerAgent:
        let tokenIndex = baseIndex + i * TokenSize
        result[tokenIndex] = 0xFF
        result[tokenIndex + 1] = 0xFF
        result[tokenIndex + 2] = 0xFF
        
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

# Action metadata functions
proc getActionNames*(): seq[string] =
  ## Get the names of all available actions
  @["NOOP", "MOVE", "ATTACK", "GET", "SWAP", "PUT"]

proc getMaxActionArgs*(): seq[int] =
  ## Get maximum argument values for each action type
  ## NOOP=0, MOVE/ATTACK/GET/PUT=0-7 (8 directions), SWAP=0-1 (inventory positions)
  @[0, 7, 7, 7, 1, 7]

proc getFeatureNormalizations*(): seq[float] =
  ## Get normalization values for each observation layer
  result = newSeq[float](ObservationLayers)
  for i in 0..<ObservationLayers:
    result[i] = 1.0  # Default normalization for all layers

# Helper procedures
proc defaultMaxSteps*(): int =
  ## Get default max steps value
  1000

# Export sequences
exportSeq seq[int]:
  discard

exportSeq seq[float]:
  discard

exportSeq seq[bool]:
  discard

exportSeq seq[string]:
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
    getTokenObservations(TribalEnv)
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
  getActionNames
  getMaxActionArgs
  getFeatureNormalizations

# Generate the Python binding files and include implementation (must be at the end)
writeFiles("bindings/generated", "Tribal")
include generated/internal