import std/[json],
  vmath

type
  ItemAmount* = object
    itemId: int
    count: int

  Entity* = ref object
    # Common keys.
    id: int
    typeId: int
    groupId: int
    agentId: int
    location: seq[IVec3]
    orientation: int
    inventory: seq[seq[ItemAmount]]
    inventoryMax: int
    color: int

    # Agent specific keys.
    actionId: seq[int]
    actionParameter: seq[int]
    actionSuccess: seq[bool]
    currentReward: seq[int]
    totalReward: seq[int]
    isFrozen: seq[bool]
    frozenProgress: seq[int]
    frozenTime: seq[int]
    visionSize: int

    # Building specific keys.
    inputResources: seq[seq[ItemAmount]]
    outputResources: seq[seq[ItemAmount]]
    recipeMax: int
    productionProgress: seq[int]
    productionTime: int
    cooldownProgress: seq[int]
    cooldownTime: int

    # Computed fields.
    gainMap: seq[seq[ItemAmount]]
    isAgent: bool

type
  Replay* = ref object
    version: int
    numAgents: int
    maxSteps: int
    mapSize: (int, int)
    fileName: string
    typeNames: seq[string]
    actionNames: seq[string]
    itemNames: seq[string]
    groupNames: seq[string]
    objects: seq[Entity]
    rewardSharingMatrix: seq[seq[float]]
    agents: seq[Entity]
    envConfig: JsonNode
