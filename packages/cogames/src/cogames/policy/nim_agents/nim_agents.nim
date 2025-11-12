import
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
      rawObservation: pointer,
      numActions: int,
      agentAction: ptr int32
    ) {.raises: [].}
) {.raises: [].} =
  var ids: ptr UncheckedArray[int32] = nil
  if agentIds != nil:
    ids = cast[ptr UncheckedArray[int32]](agentIds)

  var obsArray: ptr UncheckedArray[uint8] = nil
  if rawObservations != nil:
    obsArray = cast[ptr UncheckedArray[uint8]](rawObservations)

  var actionArray: ptr UncheckedArray[int32] = nil
  if rawActions != nil:
    actionArray = cast[ptr UncheckedArray[int32]](rawActions)

  let obsStride = numTokens * sizeToken

  proc runAgent(idx: int) =
    if idx < 0 or idx >= agents.len:
      return
    var obsPtr: pointer = nil
    if not obsArray.isNil:
      obsPtr = cast[pointer](obsArray[idx * obsStride].addr)
    var actPtr: ptr int32 = nil
    if not actionArray.isNil:
      actPtr = cast[ptr int32](actionArray[idx].addr)
    stepProc(agents[idx], numAgents, numTokens, sizeToken, obsPtr, numActions, actPtr)

  if numAgentIds == 0:
    return  # Don't step any agents
  elif agentIds != nil and numAgentIds > 0:
    for i in 0 ..< numAgentIds:
      runAgent(ids[i])
  else:
    for idx in 0 ..< agents.len:
      runAgent(idx)

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
  if agentId < 0 or agentId >= policy.agents.len:
    return
  var actionPtrValue: ptr int32 = nil
  if rawAction != nil:
    actionPtrValue = cast[ptr int32](rawAction)
  random_agents.step(policy.agents[agentId], numAgents, numTokens, sizeToken, rawObservation, numActions, actionPtrValue)

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

proc thinkyPolicyStepSingle*(
    policy: ThinkyPolicy,
    agentId: int,
    numAgents: int,
    numTokens: int,
    sizeToken: int,
    rawObservation: pointer,
    numActions: int,
    rawAction: pointer
) {.raises: [].} =
  if agentId < 0 or agentId >= policy.agents.len:
    return
  var actionPtrValue: ptr int32 = nil
  if rawAction != nil:
    actionPtrValue = cast[ptr int32](rawAction)
  thinky_agents.step(policy.agents[agentId], numAgents, numTokens, sizeToken, rawObservation, numActions, actionPtrValue)

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

proc raceCarPolicyStepSingle*(
    policy: RaceCarPolicy,
    agentId: int,
    numAgents: int,
    numTokens: int,
    sizeToken: int,
    rawObservation: pointer,
    numActions: int,
    rawAction: pointer
) {.raises: [].} =
  if agentId < 0 or agentId >= policy.agents.len:
    return
  var actionPtrValue: ptr int32 = nil
  if rawAction != nil:
    actionPtrValue = cast[ptr int32](rawAction)
  race_car_agents.step(
    policy.agents[agentId],
    numAgents,
    numTokens,
    sizeToken,
    rawObservation,
    numActions,
    actionPtrValue
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
    randomPolicyStepSingle(RandomPolicy, int, int, int, int, pointer, int, pointer)
    randomPolicyReset(RandomPolicy)

exportRefObject ThinkyPolicy:
  constructor:
    newThinkyPolicy(string)
  procs:
    thinkyPolicyStepBatch(ThinkyPolicy, pointer, int, int, int, int, pointer, int, pointer)
    thinkyPolicyStepSingle(ThinkyPolicy, int, int, int, int, pointer, int, pointer)
    thinkyPolicyReset(ThinkyPolicy)

exportRefObject RaceCarPolicy:
  constructor:
    newRaceCarPolicy(string)
  procs:
    raceCarPolicyStepBatch(RaceCarPolicy, pointer, int, int, int, int, pointer, int, pointer)
    raceCarPolicyStepSingle(RaceCarPolicy, int, int, int, int, pointer, int, pointer)
    raceCarPolicyReset(RaceCarPolicy)

writeFiles("bindings/generated", "NimAgents")

include bindings/generated/internal
