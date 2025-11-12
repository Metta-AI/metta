import
  std/[tables],
  genny,
  common, random_agents, thinky_agents, race_car_agents

proc ctrlCHandler() {.noconv.} =
  echo "\nNim DLL caught ctrl-c, exiting..."
  quit(0)

proc nim_agents_init_chook*() =
  setControlCHook(ctrlCHandler)
  echo "NimAgents initialized"

type
  RandomPolicy* = ref object
    agents*: seq[RandomAgent]
  ThinkyPolicy* = ref object
    agents*: seq[ThinkyAgent]
  RaceCarPolicy* = ref object
    agents*: seq[RaceCarAgent]

proc newRandomPolicy*(environmentConfig: string): RandomPolicy {.raises: [].} =
  let cfg = parseConfig(environmentConfig)
  result = RandomPolicy(agents: @[])
  for id in 0 ..< cfg.config.numAgents:
    result.agents.add(newRandomAgent(id, environmentConfig))

proc newThinkyPolicy*(environmentConfig: string): ThinkyPolicy {.raises: [].} =
  let cfg = parseConfig(environmentConfig)
  result = ThinkyPolicy(agents: @[])
  for id in 0 ..< cfg.config.numAgents:
    result.agents.add(newThinkyAgent(id, environmentConfig))

proc newRaceCarPolicy*(environmentConfig: string): RaceCarPolicy {.raises: [].} =
  let cfg = parseConfig(environmentConfig)
  result = RaceCarPolicy(agents: @[])
  for id in 0 ..< cfg.config.numAgents:
    result.agents.add(newRaceCarAgent(id, environmentConfig))

proc stepPolicyBatch[T](
    agents: seq[T],
    agentIds: pointer,
    numAgentIds: int,
    numAgents: int,
    numTokens: int,
    sizeToken: int,
    rawObservations: pointer,
    numActions: int,
    rawActions: pointer,
    stepProc: proc (
      agent: T,
      numAgents: int,
      numTokens: int,
      sizeToken: int,
      rawObservations: pointer,
      numActions: int,
      rawActions: pointer
    ) {.raises: [].}
) {.raises: [].} =
  var ids: ptr UncheckedArray[int32] = nil
  if agentIds != nil:
    ids = cast[ptr UncheckedArray[int32]](agentIds)
  if agentIds != nil and numAgentIds > 0:
    for i in 0 ..< numAgentIds:
      let idx = ids[i]
      if idx >= 0 and idx < agents.len:
        stepProc(agents[idx], numAgents, numTokens, sizeToken, rawObservations, numActions, rawActions)
  else:
    for agent in agents:
      stepProc(agent, numAgents, numTokens, sizeToken, rawObservations, numActions, rawActions)

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
    random_agents.step
  )

proc thinkyPolicyStepBatch*(
    policy: ThinkyPolicy,
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
    thinky_agents.step
  )

proc raceCarPolicyStepBatch*(
    policy: RaceCarPolicy,
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
    race_car_agents.step
  )

exportProcs:
  nim_agents_init_chook

exportRefObject RandomPolicy:
  constructor:
    newRandomPolicy(string)
  procs:
    randomPolicyStepBatch(RandomPolicy, pointer, int, int, int, int, pointer, int, pointer)

exportRefObject ThinkyPolicy:
  constructor:
    newThinkyPolicy(string)
  procs:
    thinkyPolicyStepBatch(ThinkyPolicy, pointer, int, int, int, int, pointer, int, pointer)

exportRefObject RaceCarPolicy:
  constructor:
    newRaceCarPolicy(string)
  procs:
    raceCarPolicyStepBatch(RaceCarPolicy, pointer, int, int, int, int, pointer, int, pointer)

writeFiles("bindings/generated", "NimAgents")

include bindings/generated/internal
