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


# Environment status

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
    getCurrentStep(TribalEnv)
    isEpisodeDone(TribalEnv)
    renderText(TribalEnv)

# Environment status functions
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
  initBuiltinAIController
  initExternalNNController
  hasActiveController
  getControllerTypeString
  isEmulated
  isDone
  resetAndGetObsPointer
  stepWithPointers

# Generate the Python binding files and include implementation (must be at the end)
writeFiles("bindings/generated", "Tribal")
include generated/internal