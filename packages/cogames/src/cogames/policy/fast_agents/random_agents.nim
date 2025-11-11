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
  numAgents: int,
  numTokens: int,
  sizeToken: int,
  rawObservations: pointer,
  numActions: int,
  rawActions: pointer
) {.raises: [].} =
  let actions = cast[ptr UncheckedArray[int32]](rawActions)
  let action = agent.random.rand(1 .. 4).int32
  actions[agent.agentId] = action
  echo "  RandomAgent taking action: ", action
