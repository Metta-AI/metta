import
  std/[strformat, random, strutils],
  vmath

const
  # From config
  # MapLayoutRoomsX* = 1
  # MapLayoutRoomsY* = 1
  # MapBorder* = 4
  # MapRoomWidth* = 100
  # MapRoomHeight* = 100
  # MapRoomBorder* = 0
  # MapRoomObjectsAgents* = 70
  # MapRoomObjectsAltars* = 50
  # MapRoomObjectsConverters* = 100
  # MapRoomObjectsGenerators* = 100
  # MapRoomObjectsWalls* = 400

  MapLayoutRoomsX* = 1
  MapLayoutRoomsY* = 1
  MapBorder* = 4
  MapRoomWidth* = 40
  MapRoomHeight* = 40
  MapRoomBorder* = 0
  MapRoomObjectsAgents* = 10
  MapRoomObjectsAltars* = 10
  MapRoomObjectsConverters* = 10
  MapRoomObjectsGenerators* = 10
  MapRoomObjectsWalls* = 40

  MapObjectAgentInitialEnergy* = 250
  MapObjectAgentMaxEnergy* = 250
  MapObjectAgentMaxInventory* = 5
  MapObjectAgentFreezeDuration* = 10
  MapObjectAgentEnergyReward* = 0f
  MapObjectAgentHp* = 1
  MapObjectAgentMortal* = false
  MapObjectAgentUpkeepTime* = 0
  MapObjectAgentUpkeepShield* = 1
  MapObjectAgentUseCost* = 0
  MapObjectAgentAttackDamage* = 10
  MapObjectAgentAttackCost* = 5

  MapObjectAltarHp* = 30
  MapObjectAltarCooldown* = 10
  MapObjectAltarUseCost* = 100

  MapObjectConverterHp* = 30
  MapObjectConverterCooldown* = 2
  MapObjectConverterEnergyOutput* = 100

  MapObjectGeneratorHp* = 30
  MapObjectGeneratorCooldown* = 5
  MapObjectGeneratorInitialResources* = 30
  MapObjectGeneratorUseCost* = 0

  MapObjectWallHp* = 10

  ObservationLayers* = 24
  ObservationWidth* = 11
  ObservationHeight* = 11

  # Computed
  MapAgents* = MapRoomObjectsAgents * MapLayoutRoomsX * MapLayoutRoomsY
  MapWidth* = MapLayoutRoomsX * (MapRoomWidth + MapRoomBorder) + MapBorder
  MapHeight* = MapLayoutRoomsY * (MapRoomHeight + MapRoomBorder) + MapBorder

proc ivec2*(x, y: int): IVec2 =
  ## Create a new 2D vector
  result.x = x.int32
  result.y = y.int32

type
  ObservationName* = enum
    AgentLayer = 0
    AgentHpLayer = 1
    AgentFrozenLayer = 2
    AgentEnergyLayer = 3
    AgentOrientationLayer = 4
    AgentShieldLayer = 5
    AgentInventory1Layer = 6
    AgentInventory2Layer = 7
    AgentInventory3Layer = 8
    WallLayer = 9
    WallHpLayer = 10
    GeneratorLayer = 11
    GeneratorHpLayer = 12
    GeneratorResourceLayer = 13
    GeneratorReadyLayer = 14
    ConverterLayer = 15
    ConverterHpLayer = 16
    ConverterInputResourceLayer = 17
    ConverterOutputResourceLayer = 18
    ConverterOutputEnergyLayer = 19
    ConverterReadyLayer = 20
    AltarLayer = 21
    AltarHpLayer = 22
    AltarReadyLayer = 23

  Orientation* = enum
    N # Up, Key W
    S # Down, Key S
    W # Right, Key D
    E # Left, Key A

  ThingKind* = enum
    Agent
    Wall
    Generator
    Converter
    Altar

  Thing* = ref object
    kind*: ThingKind
    pos*: IVec2
    id*: int
    layer*: int
    hp*: int

    inputResource*: int
    outputResource*: int
    outputEnergy*: int
    cooldown*: int

    # Agent:
    agentId*: int
    orientation*: Orientation
    energy*: int
    frozen*: int
    shield*: bool
    inventory*: int
    reward*: float32

  Stats* = ref object
    # Agent Stats:
    actionInvalid*: int
    actionAttack*: int
    actionAttackAgent*: int
    actionAttackAltar*: int
    actionAttackConverter*: int
    actionAttackGenerator*: int
    actionAttackWall*: int
    actionMove*: int
    actionNoop*: int
    actionRotate*: int
    actionShield*: int
    actionSwap*: int
    actionUse*: int
    actionUseGenerator*: int
    actionUseConverter*: int
    actionUseAltar*: int

  Environment* = ref object
    currentStep*: int
    things*: seq[Thing]
    agents*: seq[Thing]
    grid*: array[MapWidth, array[MapHeight, Thing]]
    observations*: array[
      MapAgents,
      array[ObservationLayers,
        array[ObservationWidth, array[ObservationHeight, uint8]]
      ]
    ]
    #rewards*: array[MapAgents, float32]
    terminated *: array[MapAgents, float32]
    truncated*: array[MapAgents, float32]
    stats: seq[Stats]

proc render*(env: Environment): string =
  ## Render the environment as a string.
  for y in 0 ..< MapHeight:
    for x in 0 ..< MapWidth:
      var cell = " "
      for thing in env.things:
        if thing.pos.x == x and thing.pos.y == y:
          case thing.kind
          of Agent:
            cell = "A"
          of Wall:
            cell = "#"
          of Generator:
            cell = "g"
          of Converter:
            cell = "c"
          of Altar:
            cell = "a"
          break
      result.add(cell)
    result.add("\n")

proc renderObservations*(env: Environment): string =
  ## Render the observations as a string
  const featureNames = [
    "agent",
    "agent:hp",
    "agent:frozen",
    "agent:energy",
    "agent:orientation",
    "agent:shield",
    "agent:inv:r1",
    "agent:inv:r2",
    "agent:inv:r3",
    "wall",
    "wall:hp",
    "generator",
    "generator:hp",
    "generator:r1",
    "generator:ready",
    "converter",
    "converter:hp",
    "converter:input_resource",
    "converter:output_resource",
    "converter:output_energy",
    "converter:ready",
    "altar",
    "altar:hp",
    "altar:ready",
  ]
  for id, obs in env.observations:
    result.add "Agent: " & $id & "\n"
    for layer in 0 ..< ObservationLayers:
      result.add "Feature " & $featureNames[layer] & " " & $layer & "\n"
      for y in 0 ..< ObservationHeight:
        for x in 0 ..< ObservationWidth:
          result.formatValue(obs[layer][x][y], "4d")
        result.add "\n"

proc clear[T](s: var openarray[T]) =
  ## Clear the entire array and set everything to 0.
  let p = cast[pointer](s[0].addr)
  zeroMem(p, s.len * sizeof(T))

proc clear[N: int, T](s: ptr array[N, T]) =
  ## Clear the entire array and set everything to 0.
  let p = cast[pointer](s[][0].addr)
  zeroMem(p, s[].len * sizeof(T))

proc updateObservations(env: Environment, agentId: int) =
  ## Update observations
  var obs = env.observations[agentId].addr
  obs.clear()

  let agent = env.agents[agentId]
  var
    gridOffset = agent.pos - ivec2(ObservationWidth div 2, ObservationHeight div 2)
    gridStart = gridOffset
    gridEnd = gridOffset + ivec2(ObservationWidth, ObservationHeight)
  if gridStart.x < 0:
    gridStart.x = 0
  if gridStart.y < 0:
    gridStart.y = 0
  if gridEnd.x > MapWidth: gridEnd.x = MapWidth
  if gridEnd.y > MapHeight: gridEnd.y = MapHeight

  for gy in gridStart.y ..< gridEnd.y:
    for gx in gridStart.x ..< gridEnd.x:
      let thing = env.grid[gx][gy]
      if thing == nil:
        continue
      let x = gx - gridOffset.x
      let y = gy - gridOffset.y

      case thing.kind
      of Agent:
        # agent
        obs[0][x][y] = 1
        # agent:hp
        obs[1][x][y] = thing.hp.uint8
        # agent:frozen
        obs[2][x][y] = thing.frozen.uint8
        # agent:energy
        obs[3][x][y] = thing.energy.uint8
        # agent:orientation
        obs[4][x][y] = thing.orientation.uint8
        # agent:shield
        obs[5][x][y] = 0 #thing.shield.uint8
        # agent:inv:r1
        obs[6][x][y] = thing.inventory.uint8
        # agent:inv:r2
        # obs[7][x][y] = 0
        # agent:inv:r3
        # obs[8][x][y] = 0

      of Wall:
        # wall
        obs[9][x][y] = 1
        # wall:hp
        obs[10][x][y] = thing.hp.uint8

      of Generator:
        # generator
        obs[11][x][y] = 1
        # generator:hp
        obs[12][x][y] = thing.hp.uint8
        # generator:r1
        obs[13][x][y] = thing.inputResource.uint8
        # generator:ready
        obs[14][x][y] = (thing.cooldown == 0).uint8

      of Converter:
        # converter
        obs[15][x][y] = 1
        # converter:hp
        obs[16][x][y] = thing.hp.uint8
        # converter:input_resource
        obs[17][x][y] = thing.inputResource.uint8
        # converter:output_resource
        obs[18][x][y] = 1.uint8 #thing.outputResource.uint8
        # converter:output_energy
        obs[19][x][y] = 100.uint8 #thing.outputEnergy.uint8
        # converter:ready
        obs[20][x][y] = (thing.cooldown == 0).uint8

      of Altar:
        # altar
        obs[21][x][y] = 1
        # altar:hp
        obs[22][x][y] = thing.hp.uint8
        # altar:ready
        obs[23][x][y] = (thing.cooldown == 0).uint8

proc updateObservations(
  env: Environment,
  layer: ObservationName,
  pos: IVec2,
  value: int
) =
  let layerId = ord(layer)
  for agentId in 0 ..< MapAgents:
    let x = pos.x - env.agents[agentId].pos.x + ObservationWidth div 2
    let y = pos.y - env.agents[agentId].pos.y + ObservationHeight div 2
    if x < 0 or x >= ObservationWidth or y < 0 or y >= ObservationHeight:
      continue
    env.observations[agentId][layerId][x][y] = value.uint8

proc getThing(env: Environment, pos: IVec2): Thing =
  ## Get the thing at a position
  if pos.x < 0 or pos.x >= MapWidth or pos.y < 0 or pos.y >= MapHeight:
    return nil
  return env.grid[pos.x][pos.y]

proc isEmpty*(env: Environment, pos: IVec2): bool =
  ## Check if a position is empty
  if pos.x < 0 or pos.x >= MapWidth or pos.y < 0 or pos.y >= MapHeight:
    return false
  return env.grid[pos.x][pos.y] == nil

proc orientationToVec*(orientation: Orientation): IVec2 =
  ## Convert orientation to a vector
  case orientation
  of N: result = ivec2(0, -1)
  of S: result = ivec2(0, 1)
  of E: result = ivec2(1, 0)
  of W: result = ivec2(-1, 0)

proc relativeLocation*(orientation: Orientation, distance, offset: int): IVec2 =
  ## Calculate a relative location based on orientation.
  if orientation == N:
    ivec2(-offset, -distance)
  elif orientation == S:
    ivec2(offset, distance)
  elif orientation == E:
    ivec2(distance, -offset)
  elif orientation == W:
    ivec2(-distance, offset)
  else:
    ivec2(0, 0)

proc noopAction(env: Environment, id: int, agent: Thing) =
  ## Do nothing
  inc env.stats[id].actionNoop

proc moveAction(env: Environment, id: int, agent: Thing, argument: int) =
  ## Move the agent
  var newPos = agent.pos
  if argument == 0:
    newPos -= orientationToVec(agent.orientation)
  elif argument == 1:
    newPos += orientationToVec(agent.orientation)
  else:
    inc env.stats[id].actionInvalid
    return
  if env.isEmpty(newPos):
    env.grid[agent.pos.x][agent.pos.y] = nil
    env.updateObservations(AgentLayer, agent.pos, 0)
    env.updateObservations(AgentHpLayer, agent.pos, 0)
    env.updateObservations(AgentFrozenLayer, agent.pos, 0)
    env.updateObservations(AgentEnergyLayer, agent.pos, 0)
    env.updateObservations(AgentOrientationLayer, agent.pos, 0)
    env.updateObservations(AgentShieldLayer, agent.pos, 0)
    env.updateObservations(AgentInventory1Layer, agent.pos, 0)
    #env.updateObservations(AgentInventory2Layer, agent.pos, 0)
    #env.updateObservations(AgentInventory3Layer, agent.pos, 0)

    agent.pos = newPos

    env.grid[agent.pos.x][agent.pos.y] = agent
    env.updateObservations(AgentLayer, agent.pos, 1)
    env.updateObservations(AgentHpLayer, agent.pos, agent.hp)
    env.updateObservations(AgentFrozenLayer, agent.pos, agent.frozen)
    env.updateObservations(AgentEnergyLayer, agent.pos, agent.energy)
    env.updateObservations(AgentOrientationLayer, agent.pos, agent.orientation.int)
    env.updateObservations(AgentShieldLayer, agent.pos, agent.shield.int)
    env.updateObservations(AgentInventory1Layer, agent.pos, agent.inventory)
    #env.updateObservations(AgentInventory2Layer, agent.pos, agent.inventory)
    #env.updateObservations(AgentInventory3Layer, agent.pos, agent.inventory)

    env.updateObservations(id)

    inc env.stats[id].actionMove
  else:
    inc env.stats[id].actionInvalid

proc rotateAction(env: Environment, id: int, agent: Thing, argument: int) =
  ## Rotate the agent
  if argument < 0 or argument > 3:
    inc env.stats[id].actionInvalid
    return
  agent.orientation = Orientation(argument)
  env.updateObservations(AgentOrientationLayer, agent.pos, argument)
  inc env.stats[id].actionRotate

proc jumpAction(env: Environment, id: int, agent: Thing) =
  ## Jump the agent
  discard

proc shieldAction(env: Environment, id: int, agent: Thing, argument: int) =
  ## Shield the agent
  if argument > 1:
    inc env.stats[id].actionInvalid
    return
  if agent.shield == true:
    agent.shield = false
  elif agent.energy >= MapObjectAgentUpkeepShield:
    agent.shield = true
  inc env.stats[id].actionShield

proc transferAction(env: Environment, id: int, agent: Thing) =
  ## Transfer resources
  discard

proc useAction(env: Environment, id: int, agent: Thing, argument: int) =
  ## Use resources
  if argument > 1:
    inc env.stats[id].actionInvalid
    return
  let usePos = agent.pos + orientationToVec(agent.orientation)
  var thing = env.getThing(usePos)
  if thing == nil:
    inc env.stats[id].actionInvalid
    return
  case thing.kind
  of Wall:
    inc env.stats[id].actionInvalid
  of Agent:
    inc env.stats[id].actionInvalid
  of Altar:
    if thing.cooldown == 0 and agent.energy >= MapObjectAltarUseCost:
      agent.reward += 1
      agent.energy -= MapObjectAltarUseCost
      env.updateObservations(AgentEnergyLayer, agent.pos, agent.energy)
      thing.cooldown = MapObjectAltarCooldown
      env.updateObservations(AltarReadyLayer, thing.pos, thing.cooldown)
      inc env.stats[id].actionUseAltar
      inc env.stats[id].actionUse
  of Generator:
    if thing.cooldown == 0 and agent.energy >= MapObjectGeneratorUseCost:
      agent.energy -= MapObjectGeneratorUseCost
      env.updateObservations(AgentEnergyLayer, agent.pos, agent.energy)
      agent.inventory += 1
      env.updateObservations(AgentInventory1Layer, agent.pos, agent.inventory)
      thing.cooldown = MapObjectGeneratorCooldown
      env.updateObservations(GeneratorReadyLayer, thing.pos, thing.cooldown)
      inc env.stats[id].actionUseGenerator
      inc env.stats[id].actionUse
  of Converter:
    if thing.cooldown == 0 and agent.inventory > 0:
      agent.inventory -= 1
      env.updateObservations(AgentInventory1Layer, agent.pos, agent.inventory)
      agent.energy += MapObjectConverterEnergyOutput
      env.updateObservations(AgentEnergyLayer, agent.pos, agent.energy)
      agent.energy = clamp(agent.energy, 0, MapObjectAgentMaxEnergy)
      env.updateObservations(AgentEnergyLayer, agent.pos, agent.energy)
      thing.cooldown = MapObjectConverterCooldown
      env.updateObservations(ConverterReadyLayer, thing.pos, thing.cooldown)
      inc env.stats[id].actionUseConverter
      inc env.stats[id].actionUse

proc attackAction*(env: Environment, id: int, agent: Thing, argument: int) =
  ## Attack
  if argument > 9:
    inc env.stats[id].actionInvalid
    return
  if agent.energy < MapObjectAgentAttackCost:
    # Can't attack not enough energy.
    inc env.stats[id].actionInvalid
    return
  else:
    agent.energy -= MapObjectAgentAttackCost
    env.updateObservations(AgentEnergyLayer, agent.pos, agent.energy)
  let
    # Calculate relative offsets using modulo and division
    distance = 1 + (argument - 1) div 3
    offset = -((argument - 1) mod 3 - 1)
    targetPos = agent.pos + relativeLocation(agent.orientation, distance, offset)
  var target = env.getThing(targetPos)
  if target == nil:
    return

  if target.kind != Agent:
    target.hp -= 1
    if target.hp <= 0:
      env.things.del(env.things.find(target))
      env.grid[target.pos.x][target.pos.y] = nil
    if target.kind == Altar:
      inc env.stats[id].actionAttackAltar
    elif target.kind == Converter:
      inc env.stats[id].actionAttackConverter
    elif target.kind == Generator:
      inc env.stats[id].actionAttackGenerator
    elif target.kind == Wall:
      inc env.stats[id].actionAttackWall
    inc env.stats[id].actionAttack

  elif target.kind == Agent:
    inc env.stats[id].actionAttackAgent
    inc env.stats[id].actionAttack
    if target.shield and target.energy >= MapObjectAgentAttackDamage:
      # Blocked by shield.
      target.energy -= MapObjectAgentAttackDamage
      env.updateObservations(AgentEnergyLayer, target.pos, target.energy)
    else:
      target.shield = false
      # env.updateObservations(AgentShieldLayer, target.pos, target.shield)
      target.frozen = MapObjectAgentFreezeDuration
      target.energy = 0
      env.updateObservations(AgentEnergyLayer, target.pos, target.energy)
      # Steal inventory
      if target.inventory > 0:
        target.inventory -= 1
        env.updateObservations(AgentInventory1Layer, target.pos, target.inventory)
        agent.inventory += 1
        env.updateObservations(AgentInventory1Layer, agent.pos, agent.inventory)

proc giftAction(env: Environment, id: int, agent: Thing) =
  ## Gift
  discard

proc swapAction(env: Environment, id: int, agent: Thing, argument: int) =
  ## Swap
  if argument > 1:
    inc env.stats[id].actionInvalid
    return
  let
    targetPos = agent.pos + orientationToVec(agent.orientation)
    target = env.getThing(targetPos)
  if target == nil:
    inc env.stats[id].actionInvalid
    return
  if target.kind == Agent and target.frozen > 0:
    var temp = agent.pos
    agent.pos = target.pos
    target.pos = temp
    inc env.stats[id].actionSwap
    env.updateObservations(id)
    env.updateObservations(target.agentId)
  else:
    inc env.stats[id].actionInvalid

# proc updateGrid(env: Environment) =
#   ## Update the grid
#   for x in 0 ..< MapWidth:
#     for y in 0 ..< MapHeight:
#       env.grid[x][y] = nil
#   for thing in env.things:
#     env.grid[thing.pos.x][thing.pos.y] = thing

proc randomEmptyPos(r: var Rand, env: Environment): IVec2 =
  ## Find an empty position in the environment
  for i in 0 ..< 100:
    let pos = ivec2(r.rand(0 ..< MapWidth), r.rand(0 ..< MapHeight))
    if env.isEmpty(pos):
      result = pos
      return
  quit("Failed to find an empty position, map too full!")

proc add(env: Environment, thing: Thing) =
  ## Add a thing to the environment
  env.things.add(thing)
  if thing.kind == Agent:
    env.agents.add(thing)
    env.stats.add(Stats())
  env.grid[thing.pos.x][thing.pos.y] = thing

proc init(env: Environment) =
  ## Initialize or reset the environment.

  var r = initRand(2024)

  if MapBorder > 0:
    for x in 0 ..< MapWidth:
      for j in 0 ..< MapBorder:
        env.add(Thing(kind: Wall, pos: ivec2(x, j), hp: MapObjectWallHp))
        env.add(Thing(kind: Wall, pos: ivec2(x, MapHeight - j - 1), hp: MapObjectWallHp))
    for y in 0 ..< MapHeight:
      for j in 0 ..< MapBorder:
        env.add(Thing(kind: Wall, pos: ivec2(j, y), hp: MapObjectWallHp))
        env.add(Thing(kind: Wall, pos: ivec2(MapWidth - j - 1, y), hp: MapObjectWallHp))

  for i in 0 ..< MapRoomObjectsWalls:
    let pos = r.randomEmptyPos(env)
    env.add(Thing(kind: Wall, pos: pos, hp: 10))

  for i in 0 ..< MapRoomObjectsAgents:
    let pos = r.randomEmptyPos(env)
    env.add(Thing(
      kind: Agent,
      agentId: i,
      pos: pos,
      hp: MapObjectAgentHp,
      energy: MapObjectAgentInitialEnergy,
    ))

  for i in 0 ..< MapRoomObjectsAltars:
    let pos = r.randomEmptyPos(env)
    env.add(Thing(
      kind: Altar,
      pos: pos,
      hp: MapObjectAltarHp,
    ))

  for i in 0 ..< MapRoomObjectsConverters:
    let pos = r.randomEmptyPos(env)
    env.add(Thing(
      kind: Converter,
      pos: pos,
      hp: MapObjectConverterHp,
    ))

  for i in 0 ..< MapRoomObjectsGenerators:
    let pos = r.randomEmptyPos(env)
    env.add(Thing(
      kind: Generator,
      pos: pos,
      hp: MapObjectGeneratorHp,
    ))

  for agentId in 0 ..< MapAgents:
    env.updateObservations(agentId)

proc loadMap*(env: Environment, map: string) =
  ## Load a map from a string

  env.currentStep = 0
  env.terminated.clear()
  env.truncated.clear()
  env.things.setLen(0)
  env.agents.setLen(0)
  env.stats.setLen(0)
  env.grid.clear()
  env.observations.clear()

  for line in map.split("\n"):
    let parts = line.split(" ")
    let
      kind = parseEnum[ThingKind](parts[0])
      id = parts[1].parseInt
    case kind:
    of Agent:
      env.add(Thing(
        kind: kind,
        id: id,
        agentId: parts[2].parseInt,
        pos: ivec2(parts[3].parseInt, parts[4].parseInt),
        hp: MapObjectAgentHp,
        energy: MapObjectAgentInitialEnergy,
      ))
    of Wall:
      env.add(Thing(
        kind: kind,
        id: id,
        pos: ivec2(parts[2].parseInt, parts[3].parseInt),
        hp: MapObjectWallHp,
      ))
    of Generator:
      env.add(Thing(
        kind: kind,
        id: id,
        pos: ivec2(parts[2].parseInt, parts[3].parseInt),
        hp: MapObjectGeneratorHp,
        inputResource: 30
      ))
    of Converter:
      env.add(Thing(
        kind: kind,
        id: id,
        pos: ivec2(parts[2].parseInt, parts[3].parseInt),
        hp: MapObjectConverterHp,
      ))
    of Altar:
      env.add(Thing(
        kind: kind,
        id: id,
        pos: ivec2(parts[2].parseInt, parts[3].parseInt),
        hp: MapObjectAltarHp,
      ))

  for agentId in 0 ..< MapAgents:
    env.updateObservations(agentId)

proc dumpMap*(env: Environment): string =
  ## Dump the map
  for thing in env.things:
    if thing.kind == Agent:
      result.add fmt"{thing.kind} {thing.id} {thing.agentId} {thing.pos.x} {thing.pos.y}" & "\n"
    else:
      result.add fmt"{thing.kind} {thing.id} {thing.pos.x} {thing.pos.y}" & "\n"

proc newEnvironment*(): Environment =
  ## Create a new environment
  result = Environment()
  result.init()

proc nextStep*(env: Environment, actions: ptr array[MapAgents, array[2, uint8]]) =
  ## Step the environment
  inc env.currentStep
  for id, action in actions[]:
    let agent = env.agents[id]
    if agent.frozen > 0:
      continue

    case action[0]:
    of 0: env.noopAction(id, agent)
    of 1: env.moveAction(id, agent, action[1].int)
    of 2: env.rotateAction(id, agent, action[1].int)
    of 3: env.useAction(id, agent, action[1].int)
    of 4: env.attackAction(id, agent, action[1].int)
    of 6: env.shieldAction(id, agent, action[1].int)
    of 7: env.giftAction(id, agent)
    of 8: env.swapAction(id, agent, action[1].int)
    #of: env.jumpAction(id, agent)
    #of: env.transferAction(id, agent)
    else: inc env.stats[id].actionInvalid

  # Update objects
  for thing in env.things:
    if thing.kind == Altar:
      if thing.cooldown > 0:
        thing.cooldown -= 1
        env.updateObservations(AltarReadyLayer, thing.pos, thing.cooldown)
    elif thing.kind == Converter:
      if thing.cooldown > 0:
        thing.cooldown -= 1
        env.updateObservations(ConverterReadyLayer, thing.pos, thing.cooldown)
    elif thing.kind == Generator:
      if thing.cooldown > 0:
        thing.cooldown -= 1
        env.updateObservations(GeneratorReadyLayer, thing.pos, thing.cooldown)
    elif thing.kind == Agent:
      if thing.frozen > 0:
        thing.frozen -= 1
        env.updateObservations(AgentFrozenLayer, thing.pos, thing.frozen)
      if thing.shield:
        if thing.energy <= 0:
          thing.energy = 0
          thing.shield = false
        else:
          thing.energy -= MapObjectAgentUpkeepShield
        env.updateObservations(AgentEnergyLayer, thing.pos, thing.energy)
        env.updateObservations(AgentShieldLayer, thing.pos, thing.shield.int)

  # for agentId in 0 ..< MapAgents:
  #   env.updateObservations(agentId)

proc reset*(env: Environment) =
  ## Reset the environment
  env.currentStep = 0
  env.terminated.clear()
  env.truncated.clear()
  env.things.setLen(0)
  env.agents.setLen(0)
  env.stats.setLen(0)
  env.grid.clear()
  env.observations.clear()
  env.init()

proc getEpisodeStats*(env: Environment): string =
  ## Get the episode stats
  if env.stats.len == 0:
    return
  template display(name: string, statName) =
    var
      total = 0
      min = int.high
      max = 0
    for stat in env.stats:
      total += stat.statName
      if stat.statName < min:
        min = stat.statName
      if stat.statName > max:
        max = stat.statName
    let avg = total.float32 / env.stats.len.float32
    result.formatValue(name, ">26")
    result.formatValue(total, "10d")
    result.add " "
    result.formatValue(avg, "10.2f")
    result.add " "
    result.formatValue(min, "8d")
    result.add " "
    result.formatValue(max, "8d")
    result.add "\n"

  result = "                      Stat     Total    Average      Min      Max\n"
  display "action.invalid", actionInvalid
  display "action.attack", actionAttack
  display "action.attack.agent", actionAttackAgent
  display "action.attack.altar", actionAttackAltar
  display "action.attack.converter", actionAttackConverter
  display "action.attack.generator", actionAttackGenerator
  display "action.attack.wall", actionAttackWall
  display "action.move", actionMove
  display "action.noop", actionNoop
  display "action.rotate", actionRotate
  display "action.shield", actionShield
  display "action.swap", actionSwap
  display "action.use", actionUse
  display "action.use.altar", actionUseAltar
  display "action.use.converter", actionUseConverter
  display "action.use.generator", actionUseGenerator

  return result
