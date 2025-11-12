# This file has an example of random agents that just take random actions.

import
  std/random,
  common

type
  RandomAgent* = ref object
    agentId*: int
    cfg*: Config
    random*: Rand

proc newRandomAgent*(
  agentId: int,
  environmentConfig: string
): RandomAgent =
  # echo "Creating new RandomAgent ", agentId
  var config = parseConfig(environmentConfig)
  result = RandomAgent(agentId: agentId, cfg: config)
  result.random = initRand(agentId)

proc reset*(agent: RandomAgent) =
  agent.random = initRand(agent.agentId)

proc step*(
  agent: RandomAgent,
  numAgents: int,
  numTokens: int,
  sizeToken: int,
  rawObservation: pointer,
  numActions: int,
  agentAction: ptr int32
) {.raises: [].} =
  discard numAgents
  discard numTokens
  discard sizeToken
  discard rawObservation
  discard numActions
  if agentAction.isNil:
    return
  let action = agent.random.rand(1 .. 4).int32
  agentAction[] = action
  # echo "  RandomAgent taking action: ", action
