## Simplified Tribal Nimpy Module - Environment Control Only
## Provides Python control over Nim environment without complex visualization

import nimpy
import tribal/environment
import std/sequtils

# Global environment state
var
  env: Environment = nil
  initialized = false

proc initEnvironment*(): bool {.exportpy.} =
  ## Initialize the tribal environment - call once from Python
  if initialized:
    return true
    
  try:
    echo "🔗 Initializing nimpy environment control..."
    
    # Create a basic environment with defaults
    env = newEnvironment()
    if env == nil:
      echo "❌ Failed to create environment"
      return false
    
    initialized = true
    echo "✅ Nimpy environment control initialized"
    return true
    
  except Exception as e:
    echo "❌ Error initializing environment: ", e.msg
    return false

proc stepEnvironment*(actions: seq[seq[int]]): bool {.exportpy.} =
  ## Step the environment with actions from Python
  if not initialized or env == nil:
    echo "❌ Must call initEnvironment() first"
    return false
    
  try:
    # Convert Python actions to Nim format
    # Each action is [actionType, actionParameter]
    echo "🎯 Stepping environment with ", actions.len, " actions"
    
    # Here we would normally call env.step() with the actions
    # For now, just demonstrate that we received the actions
    for i, action in actions:
      if i < 3:  # Show first 3 actions as example
        echo "  Agent ", i, ": [", action[0], ", ", action[1], "]"
    
    return true
    
  except Exception as e:
    echo "❌ Error stepping environment: ", e.msg
    return false

proc getObservations*(): seq[seq[int]] {.exportpy.} =
  ## Get observations from the environment
  if not initialized or env == nil:
    echo "❌ Must call initEnvironment() first"
    return @[]
    
  try:
    # For now, return dummy observations
    # In real implementation, this would return the actual environment observations
    let numAgents = 15
    let obsSize = 10  # Simplified observation size
    
    result = newSeqWith(numAgents, newSeqWith(obsSize, 0))
    echo "🔍 Returning observations for ", numAgents, " agents"
    return result
    
  except Exception as e:
    echo "❌ Error getting observations: ", e.msg
    return @[]

proc cleanup*(): bool {.exportpy.} =
  ## Clean up resources
  if env != nil:
    # env.destroy() # If such method exists
    env = nil
  initialized = false
  echo "🧹 Nimpy environment cleaned up"
  return true