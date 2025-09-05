import tribal_game
import vmath, std/[random, tables], 

type
  FoodItem* = enum
    NoFood = 0
    Bread = 1
    Stew = 2  # Future expansion possibility
  
  ClayOvenStructure* = object
    width*: int
    height*: int
    centerPos*: IVec2
    cooldown*: int
    maxCooldown*: int
    wheatCost*: int
    outputItem*: FoodItem
  
  FoodBuilding* = object
    width*: int
    height*: int
    layout*: seq[seq[char]]
    centerPos*: IVec2
  
  HungerState* = object
    currentHunger*: int       # Steps since last meal
    maxHunger*: int          # Max steps before starvation
    isStarving*: bool        # Whether agent is currently starving
    foodInventory*: seq[FoodItem]  # Food items in inventory
    maxFoodSlots*: int       # Max food inventory slots

const
  # ClayOven properties
  ClayOvenCooldown* = 10
  ClayOvenWheatCost* = 1
  ClayOvenSize* = 3
  
  # Hunger mechanics
  MaxHungerSteps* = 50     # Agent dies after 50 steps without eating
  MaxFoodInventory* = 3    # Max food items agent can carry
  
  # Food properties
  BreadHungerRestore* = 50  # Bread resets hunger to 0
  StewHungerRestore* = 75   # Future: stew could restore more

proc createClayOven*(): ClayOvenStructure =
  ## Create a clay oven structure (3x3)
  ## 'o' = oven center, '#' = wall, '.' = entrance
  result.width = ClayOvenSize
  result.height = ClayOvenSize
  result.centerPos = ivec2(1, 1)
  result.maxCooldown = ClayOvenCooldown
  result.cooldown = 0
  result.wheatCost = ClayOvenWheatCost
  result.outputItem = Bread

proc createFoodBuilding*(buildingType: string): FoodBuilding =
  ## Create a food building with appropriate layout
  if buildingType == "clayOven":
    result.width = ClayOvenSize
    result.height = ClayOvenSize
    result.centerPos = ivec2(1, 1)
    result.layout = @[
      @['#', '.', '#'],
      @['.', 'o', '.'],
      @['#', '#', '#']
    ]

proc getFoodBuildingElements*(building: FoodBuilding, topLeft: IVec2): tuple[
  center: IVec2,
  walls: seq[IVec2],
  entrances: seq[IVec2],
  buildingType: char
] =
  ## Get the positions of all food building elements
  result.walls = @[]
  result.entrances = @[]
  
  for y in 0 ..< building.height:
    for x in 0 ..< building.width:
      let worldPos = topLeft + ivec2(x.int32, y.int32)
      case building.layout[y][x]:
      of 'o':
        result.center = worldPos
        result.buildingType = building.layout[y][x]
      of '#':
        result.walls.add(worldPos)
      of '.':
        result.entrances.add(worldPos)
      else:
        discard  # Empty space

proc canUseClayOven*(oven: ClayOvenStructure, agentWheat: int): bool =
  ## Check if agent can use the clay oven
  return oven.cooldown == 0 and agentWheat >= oven.wheatCost

proc useClayOven*(oven: var ClayOvenStructure, agentWheat: var int): FoodItem =
  ## Use the clay oven to bake bread
  if canUseClayOven(oven, agentWheat):
    agentWheat -= oven.wheatCost
    oven.cooldown = oven.maxCooldown
    return Bread
  return NoFood

proc updateOvenCooldown*(oven: var ClayOvenStructure) =
  ## Update cooldown for clay oven
  if oven.cooldown > 0:
    oven.cooldown -= 1

# Hunger system
proc initHungerState*(maxHunger: int = MaxHungerSteps, maxSlots: int = MaxFoodInventory): HungerState =
  ## Initialize hunger tracking for an agent
  result.currentHunger = 0
  result.maxHunger = maxHunger
  result.isStarving = false
  result.foodInventory = @[]
  result.maxFoodSlots = maxSlots

proc addFoodItem*(hunger: var HungerState, item: FoodItem): bool =
  ## Add a food item to agent's inventory
  if hunger.foodInventory.len < hunger.maxFoodSlots and item != NoFood:
    hunger.foodInventory.add(item)
    return true
  return false

proc hasFoodItem*(hunger: HungerState, item: FoodItem = NoFood): bool =
  ## Check if agent has any food (or specific food if specified)
  if item == NoFood:
    return hunger.foodInventory.len > 0
  else:
    return item in hunger.foodInventory

proc eatFood*(hunger: var HungerState, preferredFood: FoodItem = NoFood): tuple[
  ate: bool,
  foodConsumed: FoodItem,
  hungerRestored: int
] =
  ## Agent eats food to reset hunger
  ## Returns whether food was eaten, what was eaten, and hunger restored
  if hunger.foodInventory.len == 0:
    return (ate: false, foodConsumed: NoFood, hungerRestored: 0)
  
  var foodToEat = NoFood
  var foodIndex = -1
  
  # Try to eat preferred food first
  if preferredFood != NoFood:
    for i, food in hunger.foodInventory:
      if food == preferredFood:
        foodToEat = food
        foodIndex = i
        break
  
  # If no preferred food found, eat first available
  if foodIndex == -1 and hunger.foodInventory.len > 0:
    foodToEat = hunger.foodInventory[0]
    foodIndex = 0
  
  if foodIndex >= 0:
    # Consume the food
    hunger.foodInventory.delete(foodIndex)
    
    # Reset hunger based on food type
    let hungerRestore = case foodToEat:
      of Bread: BreadHungerRestore
      of Stew: StewHungerRestore
      of NoFood: 0
    
    hunger.currentHunger = max(0, hunger.currentHunger - hungerRestore)
    hunger.isStarving = false
    
    return (ate: true, foodConsumed: foodToEat, hungerRestored: hungerRestore)
  
  return (ate: false, foodConsumed: NoFood, hungerRestored: 0)

proc updateHunger*(hunger: var HungerState): tuple[
  isDying: bool
] =
  ## Update hunger state each step
  ## Returns whether they're dying of starvation
  hunger.currentHunger += 1
  
  # Check if agent is starving
  if hunger.currentHunger >= hunger.maxHunger:
    hunger.isStarving = true
    return (isDying: true)
  
  return (isDying: false)

proc shouldAgentEatAutomatically*(hunger: HungerState): bool =
  ## Determine if agent should automatically eat (when close to starving)
  # Auto-eat when 80% hungry
  return hunger.currentHunger >= (hunger.maxHunger * 8 div 10) and hunger.foodInventory.len > 0

proc handleStarvation*(hunger: var HungerState, agentPos: var IVec2, homeAltar: IVec2): tuple[
  died: bool,
  respawnPos: IVec2
] =
  ## Handle agent starvation - returns if agent died and where to respawn
  if hunger.currentHunger >= hunger.maxHunger:
    # Agent dies from starvation
    hunger.currentHunger = 0  # Reset hunger after respawn
    hunger.isStarving = false
    
    # Respawn at home altar if it exists
    if homeAltar.x >= 0 and homeAltar.y >= 0:
      return (died: true, respawnPos: homeAltar)
    else:
      # No home altar, respawn at current position (shouldn't normally happen)
      return (died: true, respawnPos: agentPos)
  
  return (died: false, respawnPos: agentPos)

proc getFoodInventoryDisplay*(hunger: HungerState): string =
  ## Get a string representation of food inventory
  if hunger.foodInventory.len == 0:
    return "No food"
  
  var counts = initTable[FoodItem, int]()
  for item in hunger.foodInventory:
    counts[item] = counts.getOrDefault(item, 0) + 1
  
  result = ""
  for item, count in counts:
    if result.len > 0:
      result &= ", "
    case item:
    of Bread: result &= $count & " bread"
    of Stew: result &= $count & " stew"
    of NoFood: discard

proc getHungerStatus*(hunger: HungerState): string =
  ## Get a string describing hunger status
  let percentage = (hunger.currentHunger.float / hunger.maxHunger.float * 100).int
  
  if hunger.isStarving:
    return "STARVING!"
  elif percentage >= 80:
    return "Very Hungry (" & $hunger.currentHunger & "/" & $hunger.maxHunger & ")"
  elif percentage >= 60:
    return "Hungry (" & $hunger.currentHunger & "/" & $hunger.maxHunger & ")"  
  elif percentage >= 40:
    return "Getting Hungry (" & $hunger.currentHunger & "/" & $hunger.maxHunger & ")"
  else:
    return "Well Fed (" & $hunger.currentHunger & "/" & $hunger.maxHunger & ")"

proc shouldShowHungerWarning*(hunger: HungerState): bool =
  ## Check if we should display a hunger warning to the player
  # Show warning when 80% hungry
  return hunger.currentHunger >= (hunger.maxHunger * 8 div 10)

# Integration helpers for main game loop
proc processAgentHunger*(hunger: var HungerState, agentPos: var IVec2, 
                         homeAltar: IVec2, agentFrozen: var int): tuple[
  died: bool,
  respawned: bool,
  autoAte: bool
] =
  ## Main hunger processing for each game step
  ## Updates hunger, handles auto-eating, and manages starvation/respawn
  
  let isDying = hunger.updateHunger()
  
  # Auto-eat if very hungry and has food
  if hunger.shouldAgentEatAutomatically():
    let eatResult = hunger.eatFood()
    if eatResult.ate:
      return (died: false, respawned: false, autoAte: true)
  
  # Handle starvation
  if isDying:
    let (died, respawnPos) = hunger.handleStarvation(agentPos, homeAltar)
    if died:
      agentPos = respawnPos
      agentFrozen = 10  # Brief freeze after respawn
      return (died: true, respawned: true, autoAte: false)
  
  return (died: false, respawned: false, autoAte: false)