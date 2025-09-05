import tribal_game
import vmath, std/[random, tables], 

type
  DefenseStructure* = object
    width*: int
    height*: int
    layout*: seq[seq[char]]
    centerPos*: IVec2
  
  DefenseItem* = enum
    NoDefense = 0
    Hat = 1
    Armor = 2
  
  WeavingLoomStructure* = object
    width*: int
    height*: int
    centerPos*: IVec2
    cooldown*: int
    maxCooldown*: int
    wheatCost*: int
    outputItem*: DefenseItem
  
  ArmoryStructure* = object
    width*: int
    height*: int
    centerPos*: IVec2
    cooldown*: int
    maxCooldown*: int
    oreCost*: int
    outputItem*: DefenseItem

const
  # WeavingLoom properties
  WeavingLoomCooldown* = 15
  WeavingLoomWheatCost* = 1
  WeavingLoomSize* = 3
  
  # Armory properties
  ArmoryCooldown* = 20
  ArmoryOreCost* = 2
  ArmorySize* = 4
  
  # Defense item properties
  HatDefenseValue* = 1  # Number of hits a hat can absorb
  ArmorDefenseValue* = 3  # Armor provides strong protection

proc createWeavingLoom*(): WeavingLoomStructure =
  ## Create a weaving loom structure (3x3)
  ## 'w' = weaving loom center, '#' = wall, '.' = entrance
  result.width = WeavingLoomSize
  result.height = WeavingLoomSize
  result.centerPos = ivec2(1, 1)
  result.maxCooldown = WeavingLoomCooldown
  result.cooldown = 0
  result.wheatCost = WeavingLoomWheatCost
  result.outputItem = Hat

proc createArmory*(): ArmoryStructure =
  ## Create an armory structure (4x4)
  ## 'A' = armory center, '#' = wall, '.' = entrance
  result.width = ArmorySize
  result.height = ArmorySize
  result.centerPos = ivec2(2, 2)
  result.maxCooldown = ArmoryCooldown
  result.cooldown = 0
  result.oreCost = ArmoryOreCost
  result.outputItem = Armor

proc createDefenseBuilding*(buildingType: string): DefenseStructure =
  ## Create a defense building with appropriate layout
  if buildingType == "weavingLoom":
    result.width = WeavingLoomSize
    result.height = WeavingLoomSize
    result.centerPos = ivec2(1, 1)
    result.layout = @[
      @['#', '.', '#'],
      @['.', 'w', '.'],
      @['#', '.', '#']
    ]
  elif buildingType == "armory":
    result.width = ArmorySize
    result.height = ArmorySize
    result.centerPos = ivec2(2, 2)
    result.layout = @[
      @['#', '#', '.', '#'],
      @['#', ' ', ' ', '#'],
      @['.', ' ', 'A', '.'],
      @['#', '#', '.', '#']
    ]

proc getDefenseBuildingElements*(building: DefenseStructure, topLeft: IVec2): tuple[
  center: IVec2,
  walls: seq[IVec2],
  entrances: seq[IVec2],
  buildingType: char
] =
  ## Get the positions of all defense building elements
  result.walls = @[]
  result.entrances = @[]
  
  for y in 0 ..< building.height:
    for x in 0 ..< building.width:
      let worldPos = topLeft + ivec2(x.int32, y.int32)
      case building.layout[y][x]:
      of 'w', 'A':
        result.center = worldPos
        result.buildingType = building.layout[y][x]
      of '#':
        result.walls.add(worldPos)
      of '.':
        result.entrances.add(worldPos)
      else:
        discard  # Empty space

proc canUseWeavingLoom*(loom: WeavingLoomStructure, agentWheat: int): bool =
  ## Check if agent can use the weaving loom
  return loom.cooldown == 0 and agentWheat >= loom.wheatCost

proc useWeavingLoom*(loom: var WeavingLoomStructure, agentWheat: var int): DefenseItem =
  ## Use the weaving loom to create a hat
  if canUseWeavingLoom(loom, agentWheat):
    agentWheat -= loom.wheatCost
    loom.cooldown = loom.maxCooldown
    return Hat
  return NoDefense

proc canUseArmory*(armory: ArmoryStructure, agentOre: int): bool =
  ## Check if agent can use the armory
  return armory.cooldown == 0 and agentOre >= armory.oreCost

proc useArmory*(armory: var ArmoryStructure, agentOre: var int): DefenseItem =
  ## Use the armory to create armor
  if canUseArmory(armory, agentOre):
    agentOre -= armory.oreCost
    armory.cooldown = armory.maxCooldown
    return Armor
  return NoDefense

proc updateDefenseCooldowns*(loom: var WeavingLoomStructure, armory: var ArmoryStructure) =
  ## Update cooldowns for defense buildings
  if loom.cooldown > 0:
    loom.cooldown -= 1
  if armory.cooldown > 0:
    armory.cooldown -= 1

# Agent defense system
type
  AgentDefense* = object
    defenseItems*: seq[DefenseItem]
    maxDefenseSlots*: int

proc initAgentDefense*(maxSlots: int = 3): AgentDefense =
  ## Initialize agent defense inventory
  result.defenseItems = @[]
  result.maxDefenseSlots = maxSlots

proc addDefenseItem*(defense: var AgentDefense, item: DefenseItem): bool =
  ## Add a defense item to agent's inventory
  if defense.defenseItems.len < defense.maxDefenseSlots and item != NoDefense:
    defense.defenseItems.add(item)
    return true
  return false

proc hasDefenseItem*(defense: AgentDefense, item: DefenseItem): bool =
  ## Check if agent has a specific defense item
  return item in defense.defenseItems

proc consumeDefenseItem*(defense: var AgentDefense, preferredItem: DefenseItem = NoDefense): DefenseItem =
  ## Consume a defense item when hit by clippy
  ## Returns the consumed item or NoDefense if none available
  if defense.defenseItems.len == 0:
    return NoDefense
  
  # Try to use preferred item first
  if preferredItem != NoDefense:
    for i, item in defense.defenseItems:
      if item == preferredItem:
        result = item
        defense.defenseItems.delete(i)
        return result
  
  # Use first available item
  result = defense.defenseItems[0]
  defense.defenseItems.delete(0)
  return result

proc getDefenseValue*(item: DefenseItem): int =
  ## Get the defense value of an item (how many hits it can absorb)
  case item:
  of NoDefense: return 0
  of Hat: return HatDefenseValue
  of Armor: return ArmorDefenseValue

proc shouldAgentSurviveClippyAttack*(defense: var AgentDefense, Damage: int): tuple[
  survives: bool,
  consumedItem: DefenseItem,
  remainingDamage: int
] =
  ## Determine if agent survives a clippy attack using defense items
  ## Returns whether agent survives, what item was consumed, and remaining damage
  
  if defense.defenseItems.len == 0:
    # No defense items, agent takes full damage
    return (survives: false, consumedItem: NoDefense, remainingDamage: clippyDamage)
  
  # Try to use best defense item available
  var bestItem = NoDefense
  var bestValue = 0
  
  for item in defense.defenseItems:
    let value = getDefenseValue(item)
    if value > bestValue:
      bestValue = value
      bestItem = item
  
  if bestItem != NoDefense:
    let consumedItem = defense.consumeDefenseItem(bestItem)
    let defenseValue = getDefenseValue(consumedItem)
    let remainingDamage = max(0, Damage - defenseValue)
    
    return (
      survives: remainingDamage == 0,
      consumedItem: consumedItem,
      remainingDamage: remainingDamage
    )
  
  return (survives: false, consumedItem: NoDefense, remainingDamage: clippyDamage)

proc getDefenseInventoryDisplay*(defense: AgentDefense): string =
  ## Get a string representation of defense inventory
  if defense.defenseItems.len == 0:
    return "No defense items"
  
  var counts = initTable[DefenseItem, int]()
  for item in defense.defenseItems:
    counts[item] = counts.getOrDefault(item, 0) + 1
  
  result = ""
  for item, count in counts:
    if result.len > 0:
      result &= ", "
    case item:
    of Hat: result &= $count & " hat(s)"
    of Armor: result &= $count & " armor"
    of NoDefense: discard