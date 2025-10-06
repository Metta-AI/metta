import std/[json, strutils],
  boxy, fidget2/[hybridrender],
  zippy, vmath, jsony

type
  ItemAmount* = object
    itemId*: int
    count*: int

  Entity* = ref object
    # Common keys.
    id*: int
    typeId*: int
    groupId*: int
    agentId*: int
    location*: seq[IVec3]
    orientation*: seq[int]
    inventory*: seq[seq[ItemAmount]]
    inventoryMax*: int
    color*: seq[int]

    # Agent specific keys.
    actionId*: seq[int]
    actionSuccess*: seq[bool]
    currentReward*: seq[float]
    totalReward*: seq[float]
    isFrozen*: seq[bool]
    frozenProgress*: seq[int]
    frozenTime*: int
    visionSize*: int

    # Building specific keys.
    inputResources*: seq[ItemAmount]
    outputResources*: seq[ItemAmount]
    recipeMax*: int
    productionProgress*: seq[int]
    productionTime*: int
    cooldownProgress*: seq[int]
    cooldownTime*: int

    # Assembler specific keys.
    cooldownRemaining*: seq[int]
    cooldownDuration*: int
    isClipped*: seq[bool]
    isClipImmune*: seq[bool]
    usesCount*: seq[int]
    maxUses*: int
    allowPartialUsage*: bool
    exhaustion*: seq[bool]
    cooldownMultiplier*: seq[float]
    currentRecipeId*: int
    recipes*: seq[RecipeInfo]

    # Computed fields.
    gainMap*: seq[seq[ItemAmount]]
    isAgent*: bool

  Replay* = ref object
    version*: int
    numAgents*: int
    maxSteps*: int
    mapSize*: (int, int)
    fileName*: string
    typeNames*: seq[string]
    actionNames*: seq[string]
    itemNames*: seq[string]
    groupNames*: seq[string]
    typeImages*: seq[string]
    actionImages*: seq[string]
    actionAttackImages*: seq[string]
    actionIconImages*: seq[string]
    itemImages*: seq[string]
    traceImages*: seq[string]
    objects*: seq[Entity]
    rewardSharingMatrix*: seq[seq[float]]

    agents*: seq[Entity]

    drawnAgentActionMask*: uint64
    mgConfig*: JsonNode


  ReplayEntity* = ref object
    ## Replay entity does not have time series and only has the current step value.
    id*: int
    typeId*: int
    groupId*: int
    agentId*: int
    location*: IVec3
    orientation*: int
    inventory*: seq[ItemAmount]
    inventoryMax*: int
    color*: int

    # Agent specific keys.
    actionId*: int
    actionSuccess*: bool
    currentReward*: float
    totalReward*: float
    isFrozen*: bool
    frozenProgress*: int
    frozenTime*: int
    visionSize*: int

    # Building specific keys.
    inputResources*: seq[ItemAmount]
    outputResources*: seq[ItemAmount]
    recipeMax*: int
    productionProgress*: int
    productionTime*: int
    cooldownProgress*: int
    cooldownTime*: int

    # Assembler specific keys.
    cooldownRemaining*: int
    cooldownDuration*: int
    isClipped*: bool
    isClipImmune*: bool
    usesCount*: int
    maxUses*: int
    allowPartialUsage*: bool
    recipes*: seq[RecipeInfo]

  RecipeInfo* = object
    pattern*: int
    inputs*: seq[ItemAmount]
    outputs*: seq[ItemAmount]
    cooldown*: int

  ReplayStep* = ref object
    step*: int
    objects*: seq[ReplayEntity]

proc parseHook*(s: string, i: var int, v: var IVec3) =
  var arr: array[3, int32]
  parseHook(s, i, arr)
  v = ivec3(arr[0], arr[1], arr[2])

proc parseHook*(s: string, i: var int, v: var ItemAmount) =
  var arr: array[2, int32]
  parseHook(s, i, arr)
  v = ItemAmount(itemId: arr[0], count: arr[1])

proc expand[T](data: JsonNode, numSteps: int, defaultValue: T): seq[T] =
  if data == nil:
    # Use the default value.
    return @[defaultValue]
  elif data.kind == JArray:
    # For coordinates, we need to expand the sequence.
    if (data.len == 0):
      return @[defaultValue]
    elif (data[0].kind != JArray):
      # Its just a single array like value.
      return @[data.to(T)]
    else:
      # Expand the sequence.
      # A sequence of pairs is expanded to a sequence of values.
      var i = 0
      var j = 0
      var v: T = defaultValue
      for i in 0 ..< numSteps:
        if j < data.len and data[j].kind == JArray and data[j][0].kind ==
            JInt and data[j][0].getInt == i:
          v = data[j][1].to(T)
          j += 1
        result.add(v)
  else:
    # A single value is a valid sequence.
    return @[data.to(T)]

let drawnAgentActionNames =
  ["attack_nearest", "put_items", "get_items", "swap"]

proc expandSequenceV2(sequence: JsonNode, numSteps: int): JsonNode =
  ## Expand an array of [step, value] pairs into an array of values per step.
  if sequence.kind != JArray:
    return sequence
  var expanded = newJArray()
  var j = 0
  var v: JsonNode = newJNull()
  for i in 0 ..< numSteps:
    if j < sequence.len and sequence[j].kind == JArray and sequence[j][0].kind == JInt and
        sequence[j][0].getInt == i:
      v = sequence[j][1]
      inc j
    expanded.add(v)
  return expanded

proc getAttrV1(obj: JsonNode, attr: string, atStep: int, defaultValue: JsonNode): JsonNode =
  ## Gets an attribute from a grid object, respecting the current step.
  if not (attr in obj):
    return defaultValue
  let prop = obj[attr]
  if prop.kind != JArray:
    return prop
  # When the value is an array (already expanded per-step), index by step.
  if atStep >= 0 and atStep < prop.len:
    return prop[atStep]
  return defaultValue

proc computeGainMap(replay: Replay) =
  ## Compute gain/loss for agents.
  var items = [
    newSeq[int](replay.itemNames.len),
    newSeq[int](replay.itemNames.len)
  ]
  for agent in replay.agents:
    agent.gainMap = newSeq[seq[ItemAmount]](replay.maxSteps)

    # Gain map for step 0.
    if agent.inventory.len == 1:
      let inventory = agent.inventory[0]
      var gainMap = newSeq[ItemAmount]()
      if inventory.len > 0:
        for i in 0 ..< items[0].len:
          items[0][i] = 0
        for item in inventory:
          gainMap.add(item)
          items[0][item.itemId] = item.count
      agent.gainMap[0] = gainMap

    # Gain map for step > 1.
    for i in 1 ..< replay.maxSteps:
      var gainMap = newSeq[ItemAmount]()
      if agent.inventory.len > i:
        let inventory = agent.inventory[i]
        let n = i mod 2
        for j in 0 ..< items[n].len:
          items[n][j] = 0
        for item in inventory:
          items[n][item.itemId] = item.count
        let m = 1 - n
        for j in 0 ..< replay.itemNames.len:
          if items[n][j] != items[m][j]:
            gainMap.add(ItemAmount(itemId: j, count: items[n][j] - items[m][j]))
      agent.gainMap[i] = gainMap

proc loadReplayString*(jsonData: string, fileName: string): Replay =
  ## Load a replay from a string.
  var jsonObj = fromJson(jsonData)

  let version = jsonObj["version"].getInt
  if version != 2:
    raise newException(ValueError, "Unsupported replay version: " & $version)

  let replay = Replay(
    version: version,
    actionNames: jsonObj["action_names"].to(seq[string]),
    itemNames: jsonObj["item_names"].to(seq[string]),
    typeNames: jsonObj["type_names"].to(seq[string]),
    numAgents: jsonObj["num_agents"].getInt,
    maxSteps: jsonObj["max_steps"].getInt,
    mapSize: (jsonObj["map_size"][0].getInt, jsonObj["map_size"][1].getInt)
  )

  replay.typeImages = newSeq[string](replay.typeNames.len)
  for i in 0 ..< replay.typeNames.len:
    replay.typeImages[i] = "objects/" & replay.typeNames[i]
    if replay.typeImages[i] notin bxy:
      replay.typeImages[i] = "objects/unknown"

  replay.actionImages = newSeq[string](replay.actionNames.len)
  for i in 0 ..< replay.actionNames.len:
    replay.actionImages[i] = "actions/" & replay.actionNames[i]
    if replay.actionImages[i] notin bxy:
      replay.actionImages[i] = "actions/unknown"

  replay.actionAttackImages = newSeq[string](9)
  for i in 0 ..< 9:
    replay.actionAttackImages[i] = "actions/attack" & $(i + 1)

  replay.actionIconImages = newSeq[string](replay.actionNames.len)
  for i in 0 ..< replay.actionNames.len:
    replay.actionIconImages[i] = "actions/icons/" & replay.actionNames[i]
    if replay.actionIconImages[i] notin bxy:
      replay.actionIconImages[i] = "actions/icons/unknown"

  replay.traceImages = newSeq[string](replay.actionNames.len)
  for i in 0 ..< replay.actionNames.len:
    replay.traceImages[i] = "trace/" & replay.actionNames[i]
    if replay.traceImages[i] notin bxy:
      replay.traceImages[i] = "trace/unknown"

  replay.itemImages = newSeq[string](replay.itemNames.len)
  for i in 0 ..< replay.itemNames.len:
    replay.itemImages[i] = "resources/" & replay.itemNames[i]
    if replay.itemImages[i] notin bxy:
      replay.itemImages[i] = "resources/unknown"

  for actionName in drawnAgentActionNames:
    let idx = replay.actionNames.find(actionName)
    if idx != -1:
      replay.drawnAgentActionMask = replay.drawnAgentActionMask or (1'u64 shl idx)

  for idx, name in replay.actionNames.pairs():
    if startsWith(name, "attack_"):
      replay.drawnAgentActionMask = replay.drawnAgentActionMask or (1'u64 shl idx)

  if "file_name" in jsonObj:
    replay.fileName = jsonObj["file_name"].getStr

  if "mg_config" in jsonObj:
    replay.mgConfig = jsonObj["mg_config"]

  for obj in jsonObj["objects"]:
    let inventoryRaw = expand[seq[seq[int]]](obj["inventory"], replay.maxSteps, @[])
    var inventory: seq[seq[ItemAmount]]
    for i in 0 ..< inventoryRaw.len:
      var itemAmounts: seq[ItemAmount]
      for j in 0 ..< inventoryRaw[i].len:
        itemAmounts.add(ItemAmount(itemId: inventoryRaw[i][j][0],
            count: inventoryRaw[i][j][1]))
      inventory.add(itemAmounts)

    let locationRaw = expand[seq[int]](obj["location"], replay.maxSteps, @[0, 0, 0])
    var location: seq[IVec3]
    for i in 0 ..< locationRaw.len:
      location.add(ivec3(
        locationRaw[i][0].int32,
        locationRaw[i][1].int32,
        locationRaw[i][2].int32
      ))

    let entity = Entity(
      id: obj["id"].getInt,
      typeId: obj["type_id"].getInt,
      location: location,
      orientation: expand[int](obj["orientation"], replay.maxSteps, 0),
      inventory: inventory,
      inventoryMax: obj["inventory_max"].getInt,
      color: expand[int](obj["color"], replay.maxSteps, 0),
    )
    if "agent_id" in obj:
      entity.isAgent = true
      entity.agentId = obj["agent_id"].getInt
      entity.groupId = obj["group_id"].getInt
      entity.isFrozen = expand[bool](obj["is_frozen"], replay.maxSteps, false)
      entity.actionId = expand[int](obj["action_id"], replay.maxSteps, 0)
      entity.actionSuccess = expand[bool](obj["action_success"],
          replay.maxSteps, false)
      entity.currentReward = expand[float](obj["current_reward"],
          replay.maxSteps, 0)
      entity.totalReward = expand[float](obj["total_reward"], replay.maxSteps, 0)
      if "frozen_progress" in obj:
        entity.frozenProgress = expand[int](obj["frozen_progress"],
            replay.maxSteps, 0)
      else:
        entity.frozenProgress = @[0]
      if "frozen_time" in obj:
        entity.frozenTime = obj["frozen_time"].getInt
      else:
        entity.frozenTime = 0
      entity.visionSize = 11 # TODO Fix this

    if "input_resources" in obj:
      for pair in obj["input_resources"]:
        entity.inputResources.add(ItemAmount(itemId: pair[0].getInt,
            count: pair[1].getInt))
      for pair in obj["output_resources"]:
        entity.outputResources.add(ItemAmount(itemId: pair[0].getInt,
            count: pair[1].getInt))
      if "recipe_max" in obj:
        entity.recipeMax = obj["recipe_max"].getInt
      else:
        entity.recipeMax = 0
      if "production_progress" in obj:
        entity.productionProgress = expand[int](obj["production_progress"],
            replay.maxSteps, 0)
      else:
        entity.productionProgress = @[0]
      if "production_time" in obj:
        entity.productionTime = obj["production_time"].getInt
      else:
        entity.productionTime = 0
      if "cooldown_progress" in obj:
        entity.cooldownProgress = expand[int](obj["cooldown_progress"],
            replay.maxSteps, 0)
      else:
        entity.cooldownProgress = @[0]
      if "cooldown_time" in obj:
        entity.cooldownTime = obj["cooldown_time"].getInt
      else:
        entity.cooldownTime = 0

    replay.objects.add(entity)

    # Populate the agents field for agent entities
    if "agent_id" in obj:
      replay.agents.add(entity)

  # compute gain maps for static replays.
  computeGainMap(replay)

  return replay

proc loadReplay*(data: string, fileName: string): Replay =
  ## Load a replay from a string.
  let jsonData = zippy.uncompress(data)
  return loadReplayString(jsonData, fileName)

proc loadReplay*(fileName: string): Replay =
  ## Load a replay from a file.
  let data = readFile(fileName)
  return loadReplay(data, fileName)

proc apply*(replay: Replay, step: int, objects: seq[ReplayEntity]) =
  ## Apply a replay step to the replay.
  let agentTypeIndex = replay.typeNames.find("agent")
  for obj in objects:
    let index = obj.id - 1
    while index >= replay.objects.len:
      replay.objects.add(Entity(id: obj.id))

    let entity = replay.objects[index]
    doAssert entity.id == obj.id, "Object id mismatch"

    entity.typeId = obj.typeId
    if obj.typeId == agentTypeIndex:
      entity.isAgent = true
    entity.groupId = obj.groupId
    entity.agentId = obj.agentId
    entity.location.add(obj.location)
    entity.orientation.add(obj.orientation)
    entity.inventory.add(obj.inventory)
    entity.inventoryMax = obj.inventoryMax
    entity.color.add(obj.color)
    entity.actionId.add(obj.actionId)
    entity.actionSuccess.add(obj.actionSuccess)
    entity.currentReward.add(obj.currentReward)
    entity.totalReward.add(obj.totalReward)
    entity.isFrozen.add(obj.isFrozen)
    entity.frozenProgress.add(obj.frozenProgress)
    entity.frozenTime = obj.frozenTime
    entity.visionSize = obj.visionSize
    entity.inputResources = obj.inputResources
    entity.outputResources = obj.outputResources
    entity.recipeMax = obj.recipeMax
    entity.productionProgress.add(obj.productionProgress)
    entity.productionTime = obj.productionTime
    entity.cooldownProgress.add(obj.cooldownProgress)
    entity.cooldownTime = obj.cooldownTime

    entity.cooldownRemaining.add(obj.cooldownRemaining)
    entity.cooldownDuration = obj.cooldownDuration
    entity.isClipped.add(obj.isClipped)
    entity.isClipImmune.add(obj.isClipImmune)
    entity.usesCount.add(obj.usesCount)
    entity.maxUses = obj.maxUses
    entity.allowPartialUsage = obj.allowPartialUsage

    entity.recipes.add(obj.recipes)

  # Extend the max steps.
  replay.maxSteps = max(replay.maxSteps, step + 1)

  # Populate the agents field for agent entities
  if replay.agents.len == 0:
    for obj in replay.objects:
      if obj.typeId == agentTypeIndex:
        replay.agents.add(obj)
    doAssert replay.agents.len == replay.numAgents, "Agents and numAgents mismatch"

  computeGainMap(replay)

proc apply*(replay: Replay, replayStepJsonData: string) =
  ## Apply a replay step to the replay.
  let replayStep = fromJson(replayStepJsonData, ReplayStep)
  replay.apply(replayStep.step, replayStep.objects)
