import std/[json],
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
    actionParameter*: seq[int]
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
    itemImages*: seq[string]
    traceImages*: seq[string]
    objects*: seq[Entity]
    rewardSharingMatrix*: seq[seq[float]]
    agents*: seq[Entity]
    mgConfig*: JsonNode

proc expand[T](data: any, numSteps: int, defaultValue: T): seq[T] =
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


proc loadReplay*(data: string, fileName: string): Replay =
  ## Load a replay from a string.

  # Decompress with zippy deflate:
  let jsonData = zippy.uncompress(data)
  let jsonObj = fromJson(jsonData)

  doAssert jsonObj["version"].getInt == 2

  let replay = Replay(
    version: jsonObj["version"].getInt,
    actionNames: jsonObj["action_names"].to(seq[string]),
    itemNames: jsonObj["item_names"].to(seq[string]),
    typeNames: jsonObj["type_names"].to(seq[string]),
    numAgents: jsonObj["num_agents"].getInt,
    maxSteps: jsonObj["max_steps"].getInt,
    mapSize: (jsonObj["map_size"][0].getInt, jsonObj["map_size"][1].getInt)
  )

  replay.traceImages = newSeq[string](replay.actionNames.len)
  for i in 0 ..< replay.actionNames.len:
    replay.traceImages[i] = "trace/" & replay.actionNames[i]

  replay.itemImages = newSeq[string](replay.itemNames.len)
  for i in 0 ..< replay.itemNames.len:
    replay.itemImages[i] = "resources/" & replay.itemNames[i]

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
      entity.actionParameter = expand[int](obj["action_param"], replay.maxSteps, 0)
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

  # Compute gain/loss for agents.
  var items = [
    newSeq[int](replay.itemNames.len),
    newSeq[int](replay.itemNames.len)
  ]
  for agent in replay.agents:
    agent.gainMap = newSeq[seq[ItemAmount]](replay.maxSteps)

    # Gain map for step 0.
    let inventory = agent.inventory[0]
    var gainMap = newSeq[ItemAmount]()
    for i in 0 ..< items[0].len:
      items[0][i] = 0
    for item in inventory:
      gainMap.add(item)
      items[0][item.itemId] = item.count
    agent.gainMap[0] = gainMap

    # Gain map for step > 1.
    for i in 1 ..< replay.maxSteps:
      let inventory = agent.inventory[i]
      var gainMap = newSeq[ItemAmount]()
      let n = i mod 2
      for j in 0 ..< items[n].len:
        items[n][j] = 0
      for item in inventory:
        items[n][item.itemId] = item.count
      let m = 1 - n
      for j in 0..<replay.itemNames.len:
        if items[n][j] != items[m][j]:
          gainMap.add(ItemAmount(itemId: j, count: items[n][j] - items[m][j]))
      agent.gainMap[i] = gainMap

  return replay

proc loadReplay*(fileName: string): Replay =
  let data = readFile(fileName)
  return loadReplay(data, fileName)
