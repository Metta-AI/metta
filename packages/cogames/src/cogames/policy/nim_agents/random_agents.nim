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

  RandomPolicy* = ref object
    agents*: seq[RandomAgent]

proc newRandomAgent*(
  agentId: int,
  environmentConfig: string
): RandomAgent =
  echo "Creating new RandomAgent ", agentId
  var config = parseConfig(environmentConfig)
  result = RandomAgent(agentId: agentId, cfg: config)
  result.random = initRand(agentId)

proc step*(
  agent: RandomAgent,
  numAgents: int,
  numTokens: int,
  sizeToken: int,
  rawObservations: pointer,
  numActions: int,
  rawActions: pointer
) =
  let actions = cast[ptr UncheckedArray[int32]](rawActions)
  let action = agent.random.rand(1 .. 4).int32
  actions[agent.agentId] = action
  echo "  RandomAgent taking action: ", action

proc stepPolicyBatch*(
    agents: seq[RandomAgent],
    agentIds: pointer,
    numAgentIds: int,
    numAgents: int,
    numTokens: int,
    sizeToken: int,
    rawObservations: pointer,
    numActions: int,
    rawActions: pointer
) =
  discard

proc newRandomPolicy*(environmentConfig: string): RandomPolicy =
  let cfg = parseConfig(environmentConfig)
  var agents: seq[RandomAgent] = @[]
  for id in 0 ..< cfg.config.numAgents:
    agents.add(newRandomAgent(id, environmentConfig))
  RandomPolicy(agents: agents)

proc randomPolicyStepBatch*(
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
  stepPolicyBatch(policy.agents, agentIds, numAgentIds, numAgents, numTokens, sizeToken, rawObservations, numActions, rawActions)
