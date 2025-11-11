# This file has an example of random agents that just take random actions.

import
  std/[strformat, strutils, tables, sets, random],
  genny, jsony,
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
  echo "Creating new RandomAgent ", agentId
  var config = parseConfig(environmentConfig)
  result = RandomAgent(agentId: agentId, cfg: config)
  result.random = initRand(agentId)

proc reset*(agent: RandomAgent) =
  echo "Resetting RandomAgent ", agent.agentId
  agent.random = initRand(agent.agentId)

proc step*(
  agent: RandomAgent,
  numTokens: int,
  sizeToken: int,
  rawObservation: pointer
): int32 {.raises: [].} =
  discard numTokens
  discard sizeToken
  discard rawObservation
  result = agent.random.rand(1 .. 4).int32
  echo "  RandomAgent taking action: ", result

proc stepBatch*(
  agent: RandomAgent,
  numAgents: int,
  numTokens: int,
  sizeToken: int,
  rawObservations: pointer,
  numActions: int,
  rawActions: pointer
) {.raises: [].} =
  discard numAgents
  discard numActions
  let observations = cast[ptr UncheckedArray[uint8]](rawObservations)
  let actions = cast[ptr UncheckedArray[int32]](rawActions)
  let agentOffset = agent.agentId * numTokens * sizeToken
  let agentObservation = cast[pointer](observations[agentOffset].addr)
  let action = step(agent, numTokens, sizeToken, agentObservation)
  actions[agent.agentId] = action
