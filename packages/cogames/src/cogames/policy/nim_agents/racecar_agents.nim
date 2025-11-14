import thinky_agents

# RaceCarPolicy now wraps the Thinky implementation so we can iterate independently while
# immediately regaining the performance baseline.

type
  RaceCarAgent* = ref object
    inner: thinky_agents.ThinkyAgent

  RaceCarPolicy* = ref object
    inner: thinky_agents.ThinkyPolicy

proc newRaceCarAgent*(agentId: int, environmentConfig: string): RaceCarAgent =
  RaceCarAgent(inner: thinky_agents.newThinkyAgent(agentId, environmentConfig))

proc newRaceCarPolicy*(environmentConfig: string): RaceCarPolicy =
  RaceCarPolicy(inner: thinky_agents.newThinkyPolicy(environmentConfig))

proc step*(
  agent: RaceCarAgent,
  numAgents: int,
  numTokens: int,
  sizeToken: int,
  rawObservation: pointer,
  numActions: int,
  agentAction: ptr int32
) =
  thinky_agents.step(agent.inner, numAgents, numTokens, sizeToken, rawObservation, numActions, agentAction)

proc stepBatch*(
    policy: RaceCarPolicy,
    agentIds: pointer,
    numAgentIds: int,
    numAgents: int,
    numTokens: int,
    sizeToken: int,
    rawObservations: pointer,
    numActions: int,
    rawActions: pointer
) =
  thinky_agents.stepBatch(policy.inner, agentIds, numAgentIds, numAgents, numTokens, sizeToken, rawObservations, numActions, rawActions)
