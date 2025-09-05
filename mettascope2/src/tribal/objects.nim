import vmath

# ============== OBJECT TYPES ==============

type
  # Simple base structure for all buildings
  BaseStructure* = object
    width*: int
    height*: int
    centerPos*: IVec2
  
  # Houses need layouts for their complex structure
  HouseStructure* = object
    width*: int
    height*: int
    centerPos*: IVec2
    layout*: seq[seq[char]]  # For walls, entrances, altar, etc.
  
  # Temple is just a simple structure
  TempleStructure* = BaseStructure
  
  # Defense structures and items
  DefenseItem* = enum
    NoDefense = 0
    Hat = 1
    Armor = 2
  
  # Production buildings have base structure + production fields
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
  
  # Food structures and items
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
  
  HungerState* = object
    currentHunger*: int       # Steps since last meal
    maxHunger*: int          # Max steps before starvation
    isStarving*: bool        # Whether agent is currently starving
    foodInventory*: seq[FoodItem]  # Food items in inventory
    maxFoodSlots*: int       # Max food inventory slots
  
  ClippyBehavior* = enum
    Patrol      # Wander around looking for targets
    Chase       # Actively pursuing a player
    Guard       # Protecting the temple
    Attack      # Engaging with player

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
  ArmoryOreCost* = 2
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
  
  # Temple properties
  TempleCooldown* = 10  # Time between Clippy spawns (doubled spawn rate)

proc createHouse*(): HouseStructure =
  ## Create a house with:
  ## - Altar in the center
  ## - Four specialized buildings in the absolute corners
  ## - Walls forming a ring with entrances at cardinal directions
  result.width = 5
  result.height = 5
  result.centerPos = ivec2(2, 2)
  
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

proc createWeavingLoom*(): WeavingLoomStructure =
  ## Create a weaving loom structure (3x3)
  ## 'w' = weaving loom center, '#' = wall, '.' = entrance
  result.width = WeavingLoomSize
  result.height = WeavingLoomSize
  result.centerPos = ivec2(1, 1)
  result.cooldown = 0
  result.maxCooldown = WeavingLoomCooldown
  result.wheatCost = WeavingLoomWheatCost
  result.outputItem = Hat

proc createArmory*(): ArmoryStructure =
  ## Create an armory structure (4x4 for higher tier defense)
  ## 'a' = armory center, '#' = wall, '.' = entrance
  result.width = ArmorySize
  result.height = ArmorySize
  result.centerPos = ivec2(2, 2)  # Offset for larger building
  result.cooldown = 0
  result.maxCooldown = ArmoryCooldown
  result.oreCost = ArmoryOreCost
  result.outputItem = Armor

proc createClayOven*(): ClayOvenStructure =
  ## Create a clay oven structure (3x3)
  ## 'o' = oven center, '#' = wall, '.' = entrance
  result.width = ClayOvenSize
  result.height = ClayOvenSize
  result.centerPos = ivec2(1, 1)
  result.cooldown = 0
  result.maxCooldown = ClayOvenCooldown
  result.wheatCost = ClayOvenWheatCost
  result.outputItem = Bread

proc createTemple*(): TempleStructure =
  ## Create a temple structure (3x3 with center as spawn point)
  result.width = 3
  result.height = 3
  result.centerPos = ivec2(1, 1)

proc initHungerState*(): HungerState =
  ## Initialize a new hunger state for an agent
  result.currentHunger = 0
  result.maxHunger = MaxHungerSteps
  result.isStarving = false
  result.foodInventory = @[]
  result.maxFoodSlots = MaxFoodInventory

# ============== HELPER FUNCTIONS ==============

proc getTempleCenter*(temple: TempleStructure, topLeft: IVec2): IVec2 =
  ## Get the world position of the temple's center (spawn point)
  return topLeft + temple.centerPos

proc shouldSpawnClippy*(templeCooldown: int, nearbyClippyCount: int): bool =
  ## Determine if a temple should spawn a new Clippy
  return templeCooldown == 0

proc getClippyBehavior*(clippy: pointer, target: pointer, distanceToTarget: float): ClippyBehavior =
  ## Determine Clippy's current behavior based on game state
  if isNil(target):
    return Patrol
  elif distanceToTarget <= 1.5:
    return Attack
  elif distanceToTarget <= ClippyVisionRange.float:
    return Chase
  else:
    return Guard