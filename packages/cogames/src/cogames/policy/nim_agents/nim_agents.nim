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

proc buildAgents[T](
    environmentConfig: string,
    factory: proc (id: int, config: string): T {.raises: [].}
): seq[T] {.raises: [].} =
  let cfg = parseConfig(environmentConfig)
  result = @[]
  for id in 0 ..< cfg.config.numAgents:
    result.add(factory(id, environmentConfig))

proc newRandomPolicy*(environmentConfig: string): RandomPolicy {.raises: [].} =
  RandomPolicy(agents: buildAgents(environmentConfig, newRandomAgent))

proc newThinkyPolicy*(environmentConfig: string): ThinkyPolicy {.raises: [].} =
  ThinkyPolicy(agents: buildAgents(environmentConfig, newThinkyAgent))

proc newRaceCarPolicy*(environmentConfig: string): RaceCarPolicy {.raises: [].} =
  RaceCarPolicy(agents: buildAgents(environmentConfig, newRaceCarAgent))

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

proc randomPolicyReset*(policy: RandomPolicy) {.raises: [].} =
  for agent in policy.agents:
    random_agents.reset(agent)

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

proc thinkyPolicyReset*(policy: ThinkyPolicy) {.raises: [].} =
  for agent in policy.agents:
    thinky_agents.reset(agent)

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

proc raceCarPolicyReset*(policy: RaceCarPolicy) {.raises: [].} =
  for agent in policy.agents:
    race_car_agents.reset(agent)

exportProcs:
  nim_agents_init_chook

exportRefObject RandomPolicy:
  constructor:
    newRandomPolicy(string)
  procs:
    randomPolicyStepBatch(RandomPolicy, pointer, int, int, int, int, pointer, int, pointer)
    randomPolicyReset(RandomPolicy)

exportRefObject ThinkyPolicy:
  constructor:
    newThinkyPolicy(string)
  procs:
    thinkyPolicyStepBatch(ThinkyPolicy, pointer, int, int, int, int, pointer, int, pointer)
    thinkyPolicyReset(ThinkyPolicy)

exportRefObject RaceCarPolicy:
  constructor:
    newRaceCarPolicy(string)
  procs:
    raceCarPolicyStepBatch(RaceCarPolicy, pointer, int, int, int, int, pointer, int, pointer)
    raceCarPolicyReset(RaceCarPolicy)

writeFiles("bindings/generated", "NimAgents")

include bindings/generated/internal
