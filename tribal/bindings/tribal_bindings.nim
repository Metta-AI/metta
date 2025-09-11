## Tribal Environment Python Bindings
## Using genny to create clean Python API for tribal environment

import genny
import ../src/tribal/environment
import ../src/tribal/external_actions

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
    
    # Create the environment and make it the global environment for the viewer
    let newEnv = newEnvironment(envConfig)
    
    # Update the global environment so the Nim viewer displays this environment
    env = newEnv
    echo "🔗 Updated global environment for Nim viewer integration"
    
    result = TribalEnv(
      env: newEnv,
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

# Controller initialization and management
proc initBuiltinAIController*(seed: int = 2024): bool =
  ## Initialize controller to use built-in AI
  try:
    initGlobalController(BuiltinAI, seed)
    return true
  except:
    lastError = getCurrentException()
    return false

# Global storage for external actions (used by callback)
var storedExternalActions: array[MapAgents, array[2, uint8]]
var hasStoredActions = false

proc externalActionCallback(): array[MapAgents, array[2, uint8]] =
  ## Callback function that returns stored external actions
  if hasStoredActions:
    hasStoredActions = false  # Mark actions as consumed
    return storedExternalActions
  else:
    # No actions available, return noop
    var noopActions: array[MapAgents, array[2, uint8]]
    for i in 0..<MapAgents:
      noopActions[i] = [0'u8, 0'u8]  # noop
    return noopActions

proc setExternalActionsFromPython*(actions: seq[int]): bool =
  ## Set external actions from Python neural network
  ## actions: flat sequence of [action_type, argument] pairs for all agents
  ## Length should be MapAgents * 2
  try:
    if actions.len != MapAgents * 2:
      return false
      
    # Convert Python actions to Nim format
    for i in 0..<MapAgents:
      let actionIndex = i * 2
      if actionIndex + 1 < actions.len:
        storedExternalActions[i][0] = actions[actionIndex].uint8
        storedExternalActions[i][1] = actions[actionIndex + 1].uint8
      else:
        storedExternalActions[i][0] = 0  # noop
        storedExternalActions[i][1] = 0
    
    hasStoredActions = true
    return true
  except:
    return false

proc initExternalNNController*(): bool =
  ## Initialize controller to use external neural network
  try:
    initGlobalController(ExternalNN)
    # Set the callback to use the stored external actions
    setExternalActionCallback(externalActionCallback)
    return true
  except:
    lastError = getCurrentException()
    return false


proc hasActiveController*(): bool =
  ## Check if any controller is active
  isExternalControllerActive() or getControllerType() == BuiltinAI

proc getControllerTypeString*(): string =
  ## Get current controller type as string
  case getControllerType():
  of BuiltinAI: "BuiltinAI"
  of ExternalNN: "ExternalNN"

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

# PufferLib compatibility functions
proc isEmulated*(): bool =
  ## Native environments are not emulated
  false

proc isDone*(tribal: TribalEnv): bool =
  ## Check if environment needs reset
  tribal.env.shouldReset

# Simplified pointer-based interface for zero-copy performance
proc resetAndGetObsPointer*(tribal: TribalEnv, obsPtrInt: int): bool =
  ## Reset environment and write token observations directly to provided memory
  ## obsPtrInt is cast from numpy array pointer of shape [MapAgents, MaxTokensPerAgent, 3]
  try:
    tribal.env.reset()
    tribal.stepCount = 0
    
    # Write observations directly to provided memory
    var index = 0
    const TokenSize = 3  # [coord_byte, layer, value]
    let obsPtr = cast[ptr uint8](obsPtrInt)
    
    for agentId in 0..<MapAgents:
      var tokenCount = 0
      
      # Convert observations to tokens for this agent
      for layer in 0..<ObservationLayers:
        for y in 0..<ObservationHeight:
          for x in 0..<ObservationWidth:
            let value = tribal.env.observations[agentId][layer][x][y]
            if value > 0 and tokenCount < MaxTokensPerAgent:
              # Pack coordinates into single byte (4 bits each, max 15)
              let coordByte = (x shl 4) or y
              
              cast[ptr uint8](cast[int](obsPtr) + index)[] = coordByte.uint8
              cast[ptr uint8](cast[int](obsPtr) + index + 1)[] = layer.uint8
              cast[ptr uint8](cast[int](obsPtr) + index + 2)[] = value.uint8
              
              index += TokenSize
              inc tokenCount
      
      # Fill remaining tokens with 0xFF (invalid marker)
      while tokenCount < MaxTokensPerAgent:
        cast[ptr uint8](cast[int](obsPtr) + index)[] = 0xFF'u8
        cast[ptr uint8](cast[int](obsPtr) + index + 1)[] = 0xFF'u8
        cast[ptr uint8](cast[int](obsPtr) + index + 2)[] = 0xFF'u8
        index += TokenSize
        inc tokenCount
        
    return true
  except:
    lastError = getCurrentException()
    return false

proc stepWithPointers*(tribal: TribalEnv, actionsPtrInt: int, obsPtrInt: int, 
                      rewardsPtrInt: int, terminalsPtrInt: int, truncationsPtrInt: int): bool =
  ## Step environment using direct pointer access for zero-copy performance
  ## All pointer arguments are cast from numpy array pointers
  ## actionsPtrInt: [MapAgents, 2] uint8 array of [action_type, argument] pairs
  ## obsPtrInt: [MapAgents, MaxTokensPerAgent, 3] uint8 array for observations  
  ## rewardsPtrInt: [MapAgents] float32 array for rewards
  ## terminalsPtrInt: [MapAgents] bool array for terminal states
  ## truncationsPtrInt: [MapAgents] bool array for truncation states
  try:
    let actionsPtr = cast[ptr uint8](actionsPtrInt)
    let obsPtr = cast[ptr uint8](obsPtrInt)
    let rewardsPtr = cast[ptr float32](rewardsPtrInt)
    let terminalsPtr = cast[ptr bool](terminalsPtrInt)
    let truncationsPtr = cast[ptr bool](truncationsPtrInt)
    
    # Read actions from pointer
    var nimActions: array[MapAgents, array[2, uint8]]
    for i in 0..<MapAgents:
      let actionIndex = i * 2
      nimActions[i][0] = cast[ptr uint8](cast[int](actionsPtr) + actionIndex)[]
      nimActions[i][1] = cast[ptr uint8](cast[int](actionsPtr) + actionIndex + 1)[]
    
    # Step the environment
    tribal.env.step(nimActions.addr)
    tribal.stepCount += 1
    
    # Write token observations directly to obsPtr
    var obsIndex = 0
    const TokenSize = 3  # [coord_byte, layer, value]
    
    for agentId in 0..<MapAgents:
      var tokenCount = 0
      
      # Convert observations to tokens for this agent
      for layer in 0..<ObservationLayers:
        for y in 0..<ObservationHeight:
          for x in 0..<ObservationWidth:
            let value = tribal.env.observations[agentId][layer][x][y]
            if value > 0 and tokenCount < MaxTokensPerAgent:
              # Pack coordinates into single byte (4 bits each, max 15)
              let coordByte = (x shl 4) or y
              
              cast[ptr uint8](cast[int](obsPtr) + obsIndex)[] = coordByte.uint8
              cast[ptr uint8](cast[int](obsPtr) + obsIndex + 1)[] = layer.uint8
              cast[ptr uint8](cast[int](obsPtr) + obsIndex + 2)[] = value.uint8
              
              obsIndex += TokenSize
              inc tokenCount
      
      # Fill remaining tokens with 0xFF (invalid marker)
      while tokenCount < MaxTokensPerAgent:
        cast[ptr uint8](cast[int](obsPtr) + obsIndex)[] = 0xFF'u8
        cast[ptr uint8](cast[int](obsPtr) + obsIndex + 1)[] = 0xFF'u8
        cast[ptr uint8](cast[int](obsPtr) + obsIndex + 2)[] = 0xFF'u8
        obsIndex += TokenSize
        inc tokenCount
    
    # Write rewards directly to rewardsPtr
    for i in 0..<MapAgents:
      cast[ptr float32](cast[int](rewardsPtr) + i * sizeof(float32))[] = 
        if i < tribal.env.agents.len: tribal.env.agents[i].reward else: 0.0
      # Reset reward after reading
      if i < tribal.env.agents.len:
        tribal.env.agents[i].reward = 0.0
    
    # Write terminals directly to terminalsPtr
    for i in 0..<MapAgents:
      cast[ptr bool](cast[int](terminalsPtr) + i * sizeof(bool))[] = tribal.env.terminated[i] != 0.0
    
    # Write truncations directly to truncationsPtr
    let isEpisodeOver = tribal.stepCount >= tribal.config.game.maxSteps
    for i in 0..<MapAgents:
      cast[ptr bool](cast[int](truncationsPtr) + i * sizeof(bool))[] = 
        isEpisodeOver or (tribal.env.truncated[i] != 0.0)
    
    return true
  except:
    lastError = getCurrentException()
    return false

# Export standalone procedures
exportProcs:
  defaultMaxSteps
  defaultTribalConfig
  getActionNames
  getMaxActionArgs
  getFeatureNormalizations
  initBuiltinAIController
  initExternalNNController
  setExternalActionsFromPython
  hasActiveController
  getControllerTypeString
  isEmulated
  isDone
  resetAndGetObsPointer
  stepWithPointers

# Generate the Python binding files and include implementation (must be at the end)
writeFiles("bindings/generated", "Tribal")
include generated/internal