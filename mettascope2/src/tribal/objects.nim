import vmath, std/tables, terrain

# Import Structure types from terrain
export terrain.Structure

# ============== OBJECT TYPES ==============

type
  
  # Defense structures and items
  DefenseItem* = enum
    NoDefense = 0
    Hat = 1
    Armor = 2
  
  # Food structures and items
  FoodItem* = enum
    NoFood = 0
    Bread = 1
    Stew = 2  # Future expansion possibility
  
  # Unified production building type with variant for resource type
  ProductionBuildingKind* = enum
    WeavingLoom
    Armory
    ClayOven
  
  ProductionBuilding* = object
    width*: int
    height*: int
    centerPos*: IVec2
    cooldown*: int
    maxCooldown*: int
    case kind*: ProductionBuildingKind
    of WeavingLoom:
      wheatCostLoom*: int
      outputDefense*: DefenseItem
    of Armory:
      oreCost*: int
      outputArmor*: DefenseItem
    of ClayOven:
      wheatCostOven*: int
      outputFood*: FoodItem
  
  
  HungerState* = object
    currentHunger*: int       # Steps since last meal
    maxHunger*: int          # Max steps before starvation
    isStarving*: bool        # Whether agent is currently starving
    foodInventory*: seq[FoodItem]  # Food items in inventory
    maxFoodSlots*: int       # Max food inventory slots
  

# ============== CONSTANTS ==============

const
  # Village
  HouseSize* = 5
  
  # WeavingLoom properties
  WeavingLoomCooldown* = 15
  WeavingLoomWheatCost* = 1
  WeavingLoomSize* = 3
  
  # Armory properties
  ArmoryCooldown* = 20
  ArmoryOreCost* = 1
  ArmorySize* = 4
  
  # Defense item properties
  HatDefenseValue* = 1  # Number of hits a hat can absorb
  ArmorDefenseValue* = 3  # Armor provides strong protection
  
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
  
  # Attack/Forge properties
  ForgeWoodCost* = 1  # Wood needed to craft a spear
  ForgeCooldown* = 5  # Cooldown after crafting
  SpearRange* = 2     # Attack range with spear (Manhattan distance)
  
  # Clippy agent properties
  ClippyAttackDamage* = 2
  ClippySpeed* = 1
  ClippyVisionRange* = 15  # Even further vision for plague-wave expansion
  ClippyWanderPriority* = 3  # How many wander steps before checking for targets
  ClippyAltarSearchRange* = 12  # Extended range for aggressive altar hunting
  ClippyAgentChaseRange* = 10  # Will chase agents within this range
  
  # Spawner properties
  # Note: SpawnerCooldown defined in environment.nim

proc createHouse*(): Structure =
  ## Create a house with:
  ## - Altar in the center
  ## - Four specialized buildings in the absolute corners
  ## - Walls forming a ring with entrances at cardinal directions
  result.width = 5
  result.height = 5
  result.centerPos = ivec2(2, 2)
  result.needsBuffer = false
  result.bufferSize = 0
  
  # Initialize the layout
  # '#' = wall, 'a' = altar, ' ' = empty space inside, '.' = entrance
  # 'A' = Armory, 'F' = Forge, 'C' = Clay Oven, 'W' = Weaving Loom
  result.layout = @[
    @['A', '#', '.', '#', 'F'],  # Top row with Armory (top-left), Forge (top-right)
    @['#', ' ', ' ', ' ', '#'],  # Second row
    @['.', ' ', 'a', ' ', '.'],  # Middle row with altar and E/W entrances
    @['#', ' ', ' ', ' ', '#'],  # Fourth row
    @['C', '#', '.', '#', 'W']   # Bottom row with Clay Oven (bottom-left), Weaving Loom (bottom-right)
  ]

proc createWeavingLoom*(): ProductionBuilding =
  ## Create a weaving loom structure (3x3)
  ## 'w' = weaving loom center, '#' = wall, '.' = entrance
  result = ProductionBuilding(kind: WeavingLoom)
  result.width = WeavingLoomSize
  result.height = WeavingLoomSize
  result.centerPos = ivec2(1, 1)
  result.cooldown = 0
  result.maxCooldown = WeavingLoomCooldown
  result.wheatCostLoom = WeavingLoomWheatCost
  result.outputDefense = Hat

proc createArmory*(): ProductionBuilding =
  ## Create an armory structure (4x4 for higher tier defense)
  ## 'a' = armory center, '#' = wall, '.' = entrance
  result = ProductionBuilding(kind: Armory)
  result.width = ArmorySize
  result.height = ArmorySize
  result.centerPos = ivec2(2, 2)  # Offset for larger building
  result.cooldown = 0
  result.maxCooldown = ArmoryCooldown
  result.oreCost = ArmoryOreCost
  result.outputArmor = Armor

proc createClayOven*(): ProductionBuilding =
  ## Create a clay oven structure (3x3)
  ## 'o' = oven center, '#' = wall, '.' = entrance
  result = ProductionBuilding(kind: ClayOven)
  result.width = ClayOvenSize
  result.height = ClayOvenSize
  result.centerPos = ivec2(1, 1)
  result.cooldown = 0
  result.maxCooldown = ClayOvenCooldown
  result.wheatCostOven = ClayOvenWheatCost
  result.outputFood = Bread

proc createSpawner*(): Structure =
  ## Create a spawner structure (3x3 with center as spawn point)
  result.width = 3
  result.height = 3
  result.centerPos = ivec2(1, 1)
  result.needsBuffer = false
  result.bufferSize = 0

proc initHungerState*(): HungerState =
  ## Initialize a new hunger state for an agent
  result.currentHunger = 0
  result.maxHunger = MaxHungerSteps
  result.isStarving = false
  result.foodInventory = @[]
  result.maxFoodSlots = MaxFoodInventory

# ============== HELPER FUNCTIONS ==============

proc getSpawnerCenter*(spawner: Structure, topLeft: IVec2): IVec2 =
  ## Get the world position of the spawner's center (spawn point)
  return topLeft + spawner.centerPos

proc shouldSpawnClippy*(spawnerCooldown: int, nearbyClippyCount: int): bool =
  ## Determine if a spawner should spawn a new Clippy
  return spawnerCooldown == 0


# ============== FOOD & HUNGER FUNCTIONS ==============

proc canUseClayOven*(oven: ProductionBuilding, agentWheat: int): bool =
  ## Check if agent can use the clay oven
  assert oven.kind == ClayOven, "This function only works with ClayOven"
  return oven.cooldown == 0 and agentWheat >= oven.wheatCostOven

proc useClayOven*(oven: var ProductionBuilding, agentWheat: var int): FoodItem =
  ## Use the clay oven to bake bread
  assert oven.kind == ClayOven, "This function only works with ClayOven"
  if canUseClayOven(oven, agentWheat):
    agentWheat -= oven.wheatCostOven
    oven.cooldown = oven.maxCooldown
    return Bread
  return NoFood

proc updateOvenCooldown*(oven: var ProductionBuilding) =
  ## Update cooldown for clay oven
  assert oven.kind == ClayOven, "This function only works with ClayOven"
  if oven.cooldown > 0:
    oven.cooldown -= 1

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

proc updateHunger*(hunger: var HungerState): tuple[isDying: bool] =
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

# Integration helper for main game loop
proc processAgentHunger*(hunger: var HungerState, agentPos: var IVec2, 
                         homeAltar: IVec2, agentFrozen: var int): tuple[
  died: bool,
  respawned: bool,
  autoAte: bool
] =
  ## Main hunger processing for each game step
  ## Updates hunger, handles auto-eating, and manages starvation/respawn
  
  let hungerResult = hunger.updateHunger()
  
  # Auto-eat if very hungry and has food
  if hunger.shouldAgentEatAutomatically():
    let eatResult = hunger.eatFood()
    if eatResult.ate:
      return (died: false, respawned: false, autoAte: true)
  
  # Handle starvation
  if hungerResult.isDying:
    let (died, respawnPos) = hunger.handleStarvation(agentPos, homeAltar)
    if died:
      agentPos = respawnPos
      agentFrozen = 10  # Brief freeze after respawn
      return (died: true, respawned: true, autoAte: false)
  
  return (died: false, respawned: false, autoAte: false)