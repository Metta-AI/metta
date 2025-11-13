# This file has an example of random agents that just take random actions.

import
  std/random,
  common,
  policy_utils

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
  let action = agent.random.rand(1 .. 4).int32
  agentAction[] = action
  # echo "  RandomAgent taking action: ", action

proc newRandomPolicy*(environmentConfig: string): RandomPolicy {.raises: [].} =
  RandomPolicy(agents: buildAgents(environmentConfig, newRandomAgent))

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
) {.raises: [].} =
  stepPolicyBatch(
    policy.agents,
    agentIds,
    numAgentIds,
    numAgents,
    numTokens,
    sizeToken,
    rawObservations,
    numActions,
    rawActions,
    step
  )

proc randomPolicyStepSingle*(
    policy: RandomPolicy,
    agentId: int,
    numAgents: int,
    numTokens: int,
    sizeToken: int,
    rawObservation: pointer,
    numActions: int,
    rawAction: pointer
) {.raises: [].} =
  let actionPtrValue = cast[ptr int32](rawAction)
  step(policy.agents[agentId], numAgents, numTokens, sizeToken, rawObservation, numActions, actionPtrValue)

proc randomPolicyReset*(policy: RandomPolicy) {.raises: [].} =
  for agent in policy.agents:
    reset(agent)
