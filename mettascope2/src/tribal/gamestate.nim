## Game state management
## Centralizes global game state for better organization

import environment

type
  GameState* = ref object
    env*: Environment
    selection*: Thing

# Global game state instance
var gameState* = GameState(
  env: nil,
  selection: nil
)

proc initGameState*() =
  ## Initialize the game state with a new environment
  gameState.env = newEnvironment()
  gameState.selection = nil

proc getEnv*(): Environment =
  ## Get the current environment
  if gameState.env == nil:
    initGameState()
  return gameState.env

proc setEnv*(env: Environment) =
  ## Set the current environment
  gameState.env = env

proc getSelection*(): Thing =
  ## Get the currently selected thing
  return gameState.selection

proc setSelection*(thing: Thing) =
  ## Set the currently selected thing
  gameState.selection = thing

proc clearSelection*() =
  ## Clear the current selection
  gameState.selection = nil

# Initialize on module load
initGameState()