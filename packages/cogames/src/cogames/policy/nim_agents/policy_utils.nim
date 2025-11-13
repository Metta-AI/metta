import common

proc buildAgents*[T](
    environmentConfig: string,
    factory: proc (id: int, config: string): T {.raises: [].}
): seq[T] {.raises: [].} =
  let cfg = parseConfig(environmentConfig)
  result = @[]
  for id in 0 ..< cfg.config.numAgents:
    result.add(factory(id, environmentConfig))

proc stepPolicyBatch*[T](
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
  let ids = cast[ptr UncheckedArray[int32]](agentIds)
  let obsArray = cast[ptr UncheckedArray[uint8]](rawObservations)
  let actionArray = cast[ptr UncheckedArray[int32]](rawActions)
  let obsStride = numTokens * sizeToken

  for i in 0 ..< numAgentIds:
    let idx = int(ids[i])
    let obsPtr = cast[pointer](obsArray[idx * obsStride].addr)
    let actPtr = cast[ptr int32](actionArray[idx].addr)
    stepProc(agents[idx], numAgents, numTokens, sizeToken, obsPtr, numActions, actPtr)
