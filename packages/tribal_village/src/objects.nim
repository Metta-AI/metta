import vmath, terrain

export terrain.Structure


type

  DefenseItem* = enum
    NoDefense = 0
    Lantern = 1
    Armor = 2

  FoodItem* = enum
    NoFood = 0
    Bread = 1
    Stew = 2  # Future expansion possibility

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
      outputLantern*: DefenseItem
    of Armory:
      oreCost*: int
      outputArmor*: DefenseItem
    of ClayOven:
      wheatCostOven*: int
      outputFood*: FoodItem



const
  HouseSize* = 5

  WeavingLoomCooldown* = 15
  WeavingLoomWheatCost* = 1
  WeavingLoomSize* = 3

  ArmoryCooldown* = 20
  ArmoryOreCost* = 1
  ArmorySize* = 4

  LanternTintRadius* = 2  # Lantern spreads tint in 5x5 area (radius 2)
  ArmorDefenseValue* = 5  # Armor provides strong protection

  ClayOvenCooldown* = 10
  ClayOvenWheatCost* = 1
  ClayOvenSize* = 3

  ForgeWoodCost* = 1  # Wood needed to craft a spear
  ForgeCooldown* = 5  # Cooldown after crafting
  SpearRange* = 2     # Attack range with spear (Manhattan distance)

  TumorAttackDamage* = 2
  TumorSpeed* = 1
  TumorVisionRange* = 15  # Even further vision for plague-wave expansion
  TumorWanderPriority* = 3  # How many wander steps before checking for targets
  TumorassemblerSearchRange* = 12  # Extended range for aggressive assembler hunting
  TumorAgentChaseRange* = 10  # Will chase agents within this range


proc createHouse*(): Structure =
  result.width = 5
  result.height = 5
  result.centerPos = ivec2(2, 2)
  result.needsBuffer = false
  result.bufferSize = 0

  result.layout = @[
    @['A', '#', '.', '#', 'F'],  # Top row with Armory (top-left), Forge (top-right)
    @['#', ' ', ' ', ' ', '#'],  # Second row
    @['.', ' ', 'a', ' ', '.'],  # Middle row with assembler and E/W entrances
    @['#', ' ', ' ', ' ', '#'],  # Fourth row
    @['C', '#', '.', '#', 'W']   # Bottom row with Clay Oven (bottom-left), Weaving Loom (bottom-right)
  ]

proc createProductionBuilding*(kind: ProductionBuildingKind): ProductionBuilding =
  result = ProductionBuilding(kind: kind)
  result.cooldown = 0

  case kind
  of WeavingLoom:
    result.width = WeavingLoomSize
    result.height = WeavingLoomSize
    result.centerPos = ivec2(1, 1)
    result.maxCooldown = WeavingLoomCooldown
    result.wheatCostLoom = WeavingLoomWheatCost
    result.outputLantern = Lantern
  of Armory:
    result.width = ArmorySize
    result.height = ArmorySize
    result.centerPos = ivec2(2, 2)
    result.maxCooldown = ArmoryCooldown
    result.oreCost = ArmoryOreCost
    result.outputArmor = Armor
  of ClayOven:
    result.width = ClayOvenSize
    result.height = ClayOvenSize
    result.centerPos = ivec2(1, 1)
    result.maxCooldown = ClayOvenCooldown
    result.wheatCostOven = ClayOvenWheatCost
    result.outputFood = Bread


proc createSpawner*(): Structure =
  result.width = 3
  result.height = 3
  result.centerPos = ivec2(1, 1)
  result.needsBuffer = false
  result.bufferSize = 0

proc useClayOven*(oven: var ProductionBuilding, agentWheat: var int): FoodItem =
  assert oven.kind == ClayOven, "This function only works with ClayOven"
  if oven.cooldown == 0 and agentWheat >= oven.wheatCostOven:
    agentWheat -= oven.wheatCostOven
    oven.cooldown = oven.maxCooldown
    return Bread
  NoFood
