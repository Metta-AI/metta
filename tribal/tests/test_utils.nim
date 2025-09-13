## Test utilities for consistent environment setup matching production
## All tests should import and use these utilities to ensure consistency

import ../src/tribal/environment
import ../src/tribal/external_actions

proc setupTestEnvironment*(): Environment =
  ## Create environment identical to production setup
  ## This matches the initialization in src/tribal/environment.nim:1482
  result = newEnvironment()

proc setupTestController*(seed: int = 42) =
  ## Initialize global controller identical to production setup
  ## This matches the initialization in src/tribal.nim and external_actions.nim
  initGlobalController(BuiltinAI, seed)

proc getTestActions*(env: Environment): array[MapAgents, array[2, uint8]] =
  ## Get actions using the same system as production
  ## This matches the behavior in src/tribal/controls.nim:9
  return getActions(env)

proc stepTestEnvironment*(env: Environment) =
  ## Step environment using production-identical method
  let actions = getTestActions(env)
  env.step(addr actions)

proc runTestSteps*(env: Environment, numSteps: int) =
  ## Run multiple steps identical to production
  for i in 0..<numSteps:
    stepTestEnvironment(env)