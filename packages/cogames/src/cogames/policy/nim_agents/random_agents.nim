# This file has an example of random agents that just take random actions.

import
  std/random,
  common

type
  RandomAgent* = ref object
    agentId*: int
    cfg*: Config
    random*: Rand

  RandomPolicy* = ref object
    agents*: seq[RandomAgent]

proc newRandomAgent*(agentId: int, environmentConfig: string): RandomAgent =
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
) =
  discard numAgents
  discard numTokens
  discard sizeToken
  discard rawObservation
  discard numActions
  agentAction[] = agent.random.rand(1 .. 4).int32

proc newRandomPolicy*(environmentConfig: string): RandomPolicy =
  let cfg = parseConfig(environmentConfig)

  var agents: seq[RandomAgent] = @[]
  for id in 0 ..< cfg.config.numAgents:
    agents.add(newRandomAgent(id, environmentConfig))
  RandomPolicy(agents: agents)

proc stepBatch*(
    policy: RandomPolicy,
    agentIds: pointer,
    numAgentIds: int,
    numAgents: int,
    numTokens: int,
    sizeToken: int,
    rawObservations: pointer,
    numActions: int,
    rawActions: pointer
) =
  let ids = cast[ptr UncheckedArray[int32]](agentIds)
  let obsArray = cast[ptr UncheckedArray[uint8]](rawObservations)
  let actionArray = cast[ptr UncheckedArray[int32]](rawActions)
  let obsStride = numTokens * sizeToken

  for i in 0 ..< numAgentIds:
    let idx = int(ids[i])
    let obsPtr = cast[pointer](obsArray[idx * obsStride].addr)
    let actPtr = cast[ptr int32](actionArray[idx].addr)
    step(policy.agents[idx], numAgents, numTokens, sizeToken, obsPtr, numActions, actPtr)
