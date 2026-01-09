import std/[json, tables, strutils],
  boxy,
  zippy, vmath, jsony,
  ./validation

type

  ActionConfig* = object
    enabled*: bool

  Protocol* = object
    minAgents*: int
    vibes*: seq[int]
    inputs*: seq[ItemAmount]
    outputs*: seq[ItemAmount]
    cooldown*: int

  RecipeInfoConfig* = tuple[pattern: seq[string], protocol: Protocol]

  ObsConfig* = object
    width*: int
    height*: int
    tokenDim*: int
    numTokens*: int
    tokenValueBase*: int

  ObjectConfig* = object
    name*: string
    typeId*: int
    mapChar*: string
    renderSymbol*: string
    tags*: seq[string]
    `type`*: string
    swappable*: bool
    recipes*: seq[RecipeInfoConfig]

  GameConfig* = object
    resourceNames*: seq[string]
    vibeNames*: seq[string]
    numAgents*: int
    maxSteps*: int
    obs*: ObsConfig
    actions*: Table[string, ActionConfig]
    objects*: Table[string, ObjectConfig]

  Config* = object
    label*: string
    game*: GameConfig
    desync_episodes*: bool

  ItemAmount* = object
    itemId*: int
    count*: int

  Entity* = ref object
    # Common keys.
    id*: int
    typeName*: string
    groupId*: int
    agentId*: int
    location*: seq[IVec2]
    orientation*: seq[int]
    inventory*: seq[seq[ItemAmount]]
    inventoryMax*: int
    color*: seq[int]
    vibeId*: seq[int]

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
    protocols*: seq[Protocol]

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
    typeImages*: Table[string, string]
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
    config*: Config

    # Cached action IDs for common actions.
    noopActionId*: int
    attackActionId*: int
    putItemsActionId*: int
    getItemsActionId*: int
    moveNorthActionId*: int
    moveSouthActionId*: int
    moveWestActionId*: int
    moveEastActionId*: int

  ReplayEntity* = ref object
    ## Replay entity does not have time series and only has the current step value.
    id*: int
    typeName*: string
    typeId*: int
    groupId*: int
    agentId*: int
    location*: IVec2
    orientation*: int
    inventory*: seq[ItemAmount]
    inventoryMax*: int
    color*: int
    vibeId*: int

    # Agent specific keys.
    actionId*: int
    actionParameter*: int
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
    protocols*: seq[Protocol]


  ReplayStep* = ref object
    step*: int
    objects*: seq[ReplayEntity]

## Empty replays is used before a real replay is loaded,
## so that we don't need to check for nil everywhere.
let EmptyReplay* = Replay(
  version: 2,
  numAgents: 0,
  maxSteps: 0,
  mapSize: (0, 0),
  fileName: "",
)

proc getInt*(obj: JsonNode, key: string, default: int = 0): int =
  ## Get an integer field from JsonNode with a default value if key is missing.
  if key in obj: obj[key].getInt else: default

proc getString*(obj: JsonNode, key: string, default: string = ""): string =
  ## Get a string field from JsonNode with a default value if key is missing.
  if key in obj: obj[key].getStr else: default

proc getFloat*(obj: JsonNode, key: string, default: float = 0.0): float =
  ## Get a float field from JsonNode with a default value if key is missing.
  ## Accepts integers and converts them to floats.
  if key in obj:
    if obj[key].kind == JFloat: obj[key].getFloat
    elif obj[key].kind == JInt: obj[key].getInt.float
    else: default
  else: default

proc getBool*(obj: JsonNode, key: string, default: bool = false): bool =
  ## Get a boolean field from JsonNode with a default value if key is missing.
  if key in obj and obj[key].kind == JBool: obj[key].getBool else: default

proc getJsonNode*(obj: JsonNode, key: string, default: JsonNode = nil): JsonNode =
  ## Get any JsonNode field with a default value if key is missing.
  if key in obj: obj[key] else: default

proc getArray*(obj: JsonNode, key: string, default: JsonNode = nil): JsonNode =
  ## Get an array JsonNode field with a default value if key is missing.
  if key in obj and obj[key].kind == JArray: obj[key] else: default

proc getJsonNodeOr*(obj: JsonNode, key1: string, key2: string, default: JsonNode = nil): JsonNode =
  ## Get JsonNode field, trying key1 first, then key2, then default.
  if key1 in obj: obj[key1]
  elif key2 in obj: obj[key2]
  else: default

proc getMapSize*(obj: JsonNode, default: (int, int) = (0, 0)): (int, int) =
  ## Get map_size [width, height] with bounds checking.
  if "map_size" in obj:
    let mapSize = obj["map_size"]
    if mapSize.kind == JArray and mapSize.len >= 2:
      let w = if mapSize[0].kind == JInt: mapSize[0].getInt else: 0
      let h = if mapSize[1].kind == JInt: mapSize[1].getInt else: 0
      return (w, h)
  default

proc parseHook*(s: string, i: var int, v: var IVec2) =
  var arr: array[2, int32]
  parseHook(s, i, arr)
  v = ivec2(arr[0], arr[1])

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
      var j = 0
      var v: T = defaultValue
      for i in 0 ..< numSteps:
        if j < data.len and data[j].kind == JArray and data[j].len >= 2 and
            data[j][0].kind == JInt and data[j][0].getInt == i:
          v = data[j][1].to(T)
          j += 1
        result.add(v)
  else:
    # A single value is a valid sequence.
    return @[data.to(T)]

proc getExpandedIntSeq*(obj: JsonNode, key: string, maxSteps: int, default: seq[int] = @[0]): seq[int] =
  ## Get an expanded integer sequence field from JsonNode with a default if key is missing.
  if key in obj: expand[int](obj[key], maxSteps, 0) else: default

let drawnAgentActionNames =
  ["attack", "attack_nearest", "put_items", "get_items", "swap"]

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

proc convertReplayV1ToV2(replayData: JsonNode): JsonNode =
  ## Converts a replay from version 1 to version 2.
  echo "Converting replay from version 1 to version 2..."
  var data = newJObject()
  data["version"] = newJInt(2)

  # action_names (with renames)
  var actionNames = newJArray()
  let actionNamesArr = getArray(replayData, "action_names")
  if actionNamesArr != nil:
    for nameNode in actionNamesArr:
      var name = nameNode.getStr
      if name == "put_recipe_items":
        name = "put_items"
      elif name == "get_output":
        name = "get_items"
      actionNames.add(newJString(name))
  data["action_names"] = actionNames

  # item_names
  let invItems = getArray(replayData, "inventory_items")
  if invItems != nil and invItems.len > 0:
    data["item_names"] = invItems
  else:
    var items = newJArray()
    for s in [
      "ore.red", "ore.blue", "ore.green", "battery", "heart", "armor", "laser", "blueprint"
    ]:
      items.add(newJString(s))
    data["item_names"] = items

  data["type_names"] = getArray(replayData, "object_types", newJArray())
  data["num_agents"] = newJInt(getInt(replayData, "num_agents", 0))
  data["max_steps"] = newJInt(getInt(replayData, "max_steps", 0))

  let maxSteps = getInt(data, "max_steps", 0)

  # Helpers
  proc pair(a, b: JsonNode): JsonNode = (result = newJArray(); result.add(a); result.add(b))

  var objects = newJArray()
  var maxX = 0
  var maxY = 0
  let gridObjectsArr = getArray(replayData, "grid_objects", newJArray())
  for gridObject in gridObjectsArr:
    # Expand position and layer series if present.
    if "c" in gridObject:
      gridObject["c"] = expandSequenceV2(gridObject["c"], maxSteps)
    if "r" in gridObject:
      gridObject["r"] = expandSequenceV2(gridObject["r"], maxSteps)

    var location = newJArray()
    for step in 0 ..< maxSteps:
      let xNode = getAttrV1(gridObject, "c", step, newJInt(0))
      let yNode = getAttrV1(gridObject, "r", step, newJInt(0))
      let x = if xNode.kind == JInt: xNode.getInt else: 0
      let y = if yNode.kind == JInt: yNode.getInt else: 0
      var double = newJArray()
      double.add(newJInt(x))
      double.add(newJInt(y))
      location.add(pair(newJInt(step), double))
      if x > maxX: maxX = x
      if y > maxY: maxY = y

    # Inventory per step.
    var inventory = newJArray()
    let itemNames = data["item_names"]
    for i in 0 ..< itemNames.len:
      let inventoryName = itemNames[i].getStr
      let invKey = "inv:" & inventoryName
      let agentInvKey = "agent:inv:" & inventoryName
      if invKey in gridObject:
        gridObject[invKey] = expandSequenceV2(gridObject[invKey], maxSteps)
      elif agentInvKey in gridObject:
        gridObject[invKey] = expandSequenceV2(gridObject[agentInvKey], maxSteps)

    for step in 0 ..< maxSteps:
      var inventoryList = newJArray()
      for i in 0 ..< itemNames.len:
        let inventoryName = itemNames[i].getStr
        let invKey = "inv:" & inventoryName
        let amountNode = getAttrV1(gridObject, invKey, step, newJInt(0))
        var amt = 0
        if amountNode.kind == JInt:
          amt = amountNode.getInt
        if amt != 0:
          var pairNode = newJArray()
          pairNode.add(newJInt(i))
          pairNode.add(newJInt(amt))
          inventoryList.add(pairNode)
      inventory.add(pair(newJInt(step), inventoryList))

    # Build v2 object.
    var obj = newJObject()
    obj["id"] = newJInt(getInt(gridObject, "id", 0))

    let typeId = getInt(gridObject, "type", -1)
    obj["type_id"] = newJInt(typeId)
    if typeId >= 0 and "object_types" in replayData and typeId < replayData["object_types"].len:
      obj["type_name"] = replayData["object_types"][typeId]
    else:
      obj["type_name"] = newJString("")
    obj["location"] = location
    obj["inventory"] = inventory
    # Ensure orientation exists; default to 0 if missing.
    if "orientation" in gridObject:
      obj["orientation"] = gridObject["orientation"]
    else:
      obj["orientation"] = newJInt(0)

    # Default inventory_max to 0 for v1 (required by Nim loader).
    obj["inventory_max"] = newJInt(0)

    # Agent-specific fields.
    if "agent_id" in gridObject:
      obj["agent_id"] = gridObject["agent_id"]

      # is_frozen can be a series or a single value; coerce to bools.
      if "agent:frozen" in gridObject:
        let frozen = gridObject["agent:frozen"]
        if frozen.kind == JArray:
          var fr = newJArray()
          for p in frozen:
            var b = false
            if p[1].kind == JBool:
              b = p[1].getBool
            elif p[1].kind == JInt:
              b = p[1].getInt != 0
            fr.add(pair(p[0], newJBool(b)))
          obj["is_frozen"] = fr
        else:
          var b = if frozen.kind == JBool: frozen.getBool else: (if frozen.kind == JInt: frozen.getInt != 0 else: false)
          obj["is_frozen"] = newJBool(b)

      # color: prefer agent color; ensure presence for loader.
      if "agent:color" in gridObject:
        obj["color"] = gridObject["agent:color"]
      elif "color" in gridObject:
        obj["color"] = gridObject["color"]
      else:
        obj["color"] = newJInt(0)

      if "action_success" in gridObject:
        obj["action_success"] = gridObject["action_success"]
      obj["group_id"] = newJInt(getInt(gridObject, "agent:group", 0))
      if "agent:orientation" in gridObject:
        obj["orientation"] = gridObject["agent:orientation"]
      if "hp" in gridObject:
        obj["hp"] = gridObject["hp"]
      if "reward" in gridObject:
        obj["current_reward"] = gridObject["reward"]
      if "total_reward" in gridObject:
        obj["total_reward"] = gridObject["total_reward"]

      # Action id/param per step from combined action.
      if "action" in gridObject:
        gridObject["action"] = expandSequenceV2(gridObject["action"], maxSteps)
      var actionId = newJArray()
      var actionParam = newJArray()
      for step in 0 ..< maxSteps:
        let action = getAttrV1(gridObject, "action", step, newJNull())
        if action.kind == JArray and action.len >= 2:
          actionId.add(pair(newJInt(step), action[0]))
          actionParam.add(pair(newJInt(step), action[1]))
      obj["action_id"] = actionId
      obj["action_param"] = actionParam

    else:
      # Non-agent: ensure color exists for loader.
      if "color" in gridObject:
        obj["color"] = gridObject["color"]
      else:
        obj["color"] = newJInt(0)

    objects.add(obj)

  data["objects"] = objects
  var mapSize = newJArray()
  mapSize.add(newJInt(maxX + 1))
  mapSize.add(newJInt(maxY + 1))
  data["map_size"] = mapSize

  var mg = newJObject()
  mg["label"] = newJString("Unlabeled Replay")
  data["mg_config"] = mg

  return data

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

proc isInventoryCompressed(inventory: JsonNode): bool =
  ## Check if inventory is already in V3 compressed format [[itemId, count], ...]
  if inventory.kind != JArray or inventory.len == 0:
    return false

  for item in inventory.getElems():
    if item.kind != JArray or item.len != 2:
      return false
    let itemId = item[0]
    let count = item[1]
    if itemId.kind != JInt or count.kind != JInt:
      return false
  return true

proc compressInventoryArray(inventory: JsonNode): JsonNode =
  ## Compress a flat inventory array [itemId, itemId, ...] to [[itemId, count], ...]
  result = newJArray()
  if inventory.kind != JArray:
    return result

  var counts: seq[int]
  for itemId in inventory.getElems():
    if itemId.kind == JInt:
      let id = itemId.getInt()
      if id < 0:
        continue
      if id >= counts.len:
        counts.setLen(id + 1)
      counts[id] += 1

  for itemId, count in counts.pairs():
    if count > 0:
      var pair = newJArray()
      pair.add(newJInt(itemId))
      pair.add(newJInt(count))
      result.add(pair)

proc convertInventoryField(obj: JsonNode, fieldName: string) =
  ## Convert a single inventory field from V2 to V3 format.
  if fieldName notin obj:
    return

  let field = obj[fieldName]
  if field.kind == JArray and field.len > 0:
    # Check if this is a time series format [[step, inventory_array], ...]
    let firstItem = field[0]
    if firstItem.kind == JArray and firstItem.len == 2:
      # Time series format: convert each inventory array if needed
      var newTimeSeries = newJArray()
      var needsConversion = false
      for item in field.getElems():
        if item.kind == JArray and item.len == 2:
          let step = item[0]
          let inventoryArray = item[1]
          if inventoryArray.kind == JArray and not isInventoryCompressed(inventoryArray):
            let compressed = compressInventoryArray(inventoryArray)
            var newItem = newJArray()
            newItem.add(step)
            newItem.add(compressed)
            newTimeSeries.add(newItem)
            needsConversion = true
          else:
            newTimeSeries.add(item)
        else:
          newTimeSeries.add(item)
      if needsConversion:
        obj[fieldName] = newTimeSeries
    else:
      # Single inventory array: convert directly if needed
      if not isInventoryCompressed(field):
        obj[fieldName] = compressInventoryArray(field)
  elif field.kind == JArray and field.len == 0:
    # Empty array stays empty
    discard

proc convertReplayV2ToV3*(replayData: JsonNode): JsonNode =
  ## Convert a V2 replay to V3 format by compressing inventory arrays.
  ## V2: inventory as [itemId, itemId, ...] (repeated IDs)
  ## V3: inventory as [[itemId, count], [itemId, count], ...] (compressed pairs)
  echo "Converting replay from version 2 to version 3..."

  # Create a deep copy of the data
  var data = replayData.copy()

  # Update version to 3
  data["version"] = newJInt(3)

  # Convert inventory fields in all objects
  if "objects" in data and data["objects"].kind == JArray:
    for obj in data["objects"].getElems():
      if obj.kind != JObject:
        continue

      # Convert inventory field
      convertInventoryField(obj, "inventory")

      # Convert recipe_input field
      convertInventoryField(obj, "recipe_input")

      # Convert recipe_output field
      convertInventoryField(obj, "recipe_output")

      # Convert input_resources and output_resources (legacy building fields)
      convertInventoryField(obj, "input_resources")

      convertInventoryField(obj, "output_resources")

  return data

proc loadReplayString*(jsonData: string, fileName: string): Replay =
  ## Load a replay from a string.
  var jsonObj = fromJson(jsonData)

  if getInt(jsonObj, "version") == 1:
    jsonObj = convertReplayV1ToV2(jsonObj)

  if getInt(jsonObj, "version") == 2:
    jsonObj = convertReplayV2ToV3(jsonObj)

  doAssert getInt(jsonObj, "version") == 3

  # Check for validation issues and log them to console
  let issues = validateReplay(jsonObj)
  if issues.len > 0:
    issues.prettyPrint()
  else:
    echo "No validation issues found"

  # Safe access to required fields with defaults.
  let version = getInt(jsonObj, "version", 3)
  let actionNamesArr = getArray(jsonObj, "action_names")
  let actionNames = if actionNamesArr != nil: actionNamesArr.to(seq[string]) else: @[]
  let itemNamesArr = getArray(jsonObj, "item_names")
  let itemNames = if itemNamesArr != nil: itemNamesArr.to(seq[string]) else: @[]
  let typeNamesArr = getArray(jsonObj, "type_names")
  let typeNames = if typeNamesArr != nil: typeNamesArr.to(seq[string]) else: @[]
  let numAgents = getInt(jsonObj, "num_agents", 0)
  let maxSteps = getInt(jsonObj, "max_steps", 0)

  let replay = Replay(
    version: version,
    actionNames: actionNames,
    itemNames: itemNames,
    typeNames: typeNames,
    numAgents: numAgents,
    maxSteps: maxSteps,
    mapSize: getMapSize(jsonObj)
  )

  for actionName in drawnAgentActionNames:
    let idx = replay.actionNames.find(actionName)
    if idx != -1:
      replay.drawnAgentActionMask = replay.drawnAgentActionMask or (1'u64 shl idx)
  replay.fileName = getString(jsonObj, "file_name")

  let mgConfig = getJsonNode(jsonObj, "mg_config")
  if mgConfig != nil:
    replay.mgConfig = mgConfig
    replay.config = fromJson($mgConfig, Config)

  let objectsArr = getArray(jsonObj, "objects", newJArray())
  for obj in objectsArr:

    var inventory: seq[seq[ItemAmount]]
    if "inventory" in obj:
      let inventoryRaw = expand[seq[seq[int]]](obj["inventory"], replay.maxSteps, @[])
      for i in 0 ..< inventoryRaw.len:
        var itemAmounts: seq[ItemAmount]
        for j in 0 ..< inventoryRaw[i].len:
          if inventoryRaw[i][j].len >= 2:
            itemAmounts.add(ItemAmount(
              itemId: inventoryRaw[i][j][0],
              count: inventoryRaw[i][j][1]
            ))
        inventory.add(itemAmounts)

    var location: seq[IVec2]
    if "location" in obj:
      let locationRaw = expand[seq[int]](obj["location"], replay.maxSteps, @[0, 0])
      for coords in locationRaw:
        if coords.len >= 2:
          location.add(ivec2(coords[0].int32, coords[1].int32))
        else:
          location.add(ivec2(0, 0))
    else:
      location = @[ivec2(0, 0)]

    var resolvedTypeName = getString(obj, "type_name", "unknown")

    if resolvedTypeName == "unknown":
      let candidateId = getInt(obj, "type_id", -1)
      if candidateId >= 0 and candidateId < replay.typeNames.len:
        resolvedTypeName = replay.typeNames[candidateId]

    let entity = Entity(
      id: obj.getInt("id", 0),
      typeName: resolvedTypeName,
      location: location,
      orientation: obj.getExpandedIntSeq("orientation", replay.maxSteps),
      inventory: inventory,
      inventoryMax: obj.getInt("inventory_max", 0),
      color: obj.getExpandedIntSeq("color", replay.maxSteps),
    )
    entity.groupId = getInt(obj, "group_id", 0)

    entity.isAgent = resolvedTypeName == "agent"
    if "agent_id" in obj:
      entity.agentId = getInt(obj, "agent_id", 0)
      let frozenField = getJsonNodeOr(obj, "frozen", "is_frozen", newJBool(false))
      entity.isFrozen = expand[bool](frozenField, replay.maxSteps, false)
      let actionIdField = getJsonNode(obj, "action_id", newJInt(0))
      entity.actionId = expand[int](actionIdField, replay.maxSteps, 0)
      let actionParamField = getJsonNodeOr(obj, "action_parameter", "action_param", newJInt(0))
      entity.actionParameter = expand[int](actionParamField, replay.maxSteps, 0)
      let actionSuccessField = getJsonNode(obj, "action_success", newJBool(false))
      entity.actionSuccess = expand[bool](actionSuccessField, replay.maxSteps, false)
      let currentRewardField = getJsonNode(obj, "current_reward", newJFloat(0.0))
      entity.currentReward = expand[float](currentRewardField, replay.maxSteps, 0)
      let totalRewardField = getJsonNode(obj, "total_reward", newJFloat(0.0))
      entity.totalReward = expand[float](totalRewardField, replay.maxSteps, 0)
      let frozenProgressField = getJsonNode(obj, "frozen_progress")
      if frozenProgressField != nil:
        entity.frozenProgress = expand[int](frozenProgressField, replay.maxSteps, 0)
      else:
        entity.frozenProgress = @[0]
      entity.frozenTime = getInt(obj, "frozen_time", 0)
      entity.visionSize = 11 # TODO Fix this

      let vibeIdField = getJsonNode(obj, "vibe_id")
      if vibeIdField != nil:
        entity.vibeId = expand[int](vibeIdField, replay.maxSteps, 0)

    if "input_resources" in obj:
      for pair in obj["input_resources"]:
        if pair.kind == JArray and pair.len >= 2:
          entity.inputResources.add(ItemAmount(
            itemId: if pair[0].kind == JInt: pair[0].getInt else: 0,
            count: if pair[1].kind == JInt: pair[1].getInt else: 0
          ))
    if "output_resources" in obj:
      for pair in obj["output_resources"]:
        if pair.kind == JArray and pair.len >= 2:
          entity.outputResources.add(ItemAmount(
            itemId: if pair[0].kind == JInt: pair[0].getInt else: 0,
            count: if pair[1].kind == JInt: pair[1].getInt else: 0
          ))
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

    if "protocols" in obj:
      entity.protocols = fromJson($(obj["protocols"]), seq[Protocol])

    replay.objects.add(entity)

    # Populate the agents field for agent entities
    if "agent_id" in obj:
      replay.agents.add(entity)

  # compute gain maps for static replays.
  computeGainMap(replay)

  # Cache common action IDs for fast lookup.
  replay.noopActionId = replay.actionNames.find("noop")
  replay.attackActionId = replay.actionNames.find("attack")
  replay.putItemsActionId = replay.actionNames.find("put_items")
  replay.getItemsActionId = replay.actionNames.find("get_items")
  replay.moveNorthActionId = replay.actionNames.find("move_north")
  replay.moveSouthActionId = replay.actionNames.find("move_south")
  replay.moveWestActionId = replay.actionNames.find("move_west")
  replay.moveEastActionId = replay.actionNames.find("move_east")

  return replay

proc loadReplay*(data: string, fileName: string): Replay =
  ## Load a replay from a string.
  if fileName.endsWith(".json"):
    return loadReplayString(data, fileName)

  if not (fileName.endsWith(".json.gz") or fileName.endsWith(".json.z")):
    # TODO: Show error to user.
    echo "Unrecognized replay extension: ", fileName
    return Replay()

  let expectedFormat =
    if fileName.endsWith(".json.gz"):
      dfGzip
    else: # fileName.endsWith(".json.z"):
      dfZlib

  let jsonData =
    try:
      zippy.uncompress(data, dataFormat = expectedFormat)
    except ZippyError:
      # TODO: Show error to user.
      echo "Error uncompressing replay: ", getCurrentExceptionMsg()
      return Replay()
  return loadReplayString(jsonData, fileName)

proc loadReplay*(fileName: string): Replay =
  ## Load a replay from a file.
  let data = readFile(fileName)
  return loadReplay(data, fileName)

proc apply*(replay: Replay, step: int, objects: seq[ReplayEntity]) =
  ## Apply a replay step to the replay.
  const agentTypeName = "agent"
  for obj in objects:
    let index = obj.id - 1
    while index >= replay.objects.len:
      replay.objects.add(Entity(id: obj.id))

    let entity = replay.objects[index]
    doAssert entity.id == obj.id, "Object id mismatch"

    var resolvedTypeName = obj.typeName
    if resolvedTypeName.len == 0 and obj.typeId >= 0 and obj.typeId < replay.typeNames.len:
      resolvedTypeName = replay.typeNames[obj.typeId]
    entity.typeName = resolvedTypeName
    entity.isAgent = resolvedTypeName == agentTypeName
    entity.groupId = obj.groupId
    entity.agentId = obj.agentId
    entity.location.add(obj.location)
    entity.orientation.add(obj.orientation)
    entity.inventory.add(obj.inventory)
    entity.inventoryMax = obj.inventoryMax
    entity.color.add(obj.color)
    entity.vibeId.add(obj.vibeId)
    entity.actionId.add(obj.actionId)
    entity.actionParameter.add(obj.actionParameter)
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
    entity.protocols = obj.protocols

  # Extend the max steps.
  replay.maxSteps = max(replay.maxSteps, step + 1)

  # Populate the agents field for agent entities
  if replay.agents.len == 0:
    for obj in replay.objects:
      if obj.typeName == agentTypeName:
        replay.agents.add(obj)
    doAssert replay.agents.len == replay.numAgents, "Agents and numAgents mismatch"

  computeGainMap(replay)

proc apply*(replay: Replay, replayStepJsonData: string) =
  ## Apply a replay step to the replay.
  let replayStep = fromJson(replayStepJsonData, ReplayStep)
  replay.apply(replayStep.step, replayStep.objects)
