## Tribal Environment Configuration
## 
## This module defines the configuration structures and defaults for the tribal environment.
## Separating this from environment.nim keeps things organized and makes it easier to modify settings.

import common

# Configuration structure for environment
type
  EnvironmentConfig* = object
    # Core game parameters
    numAgents*: int
    maxSteps*: int
    episodeTruncates*: bool
    
    # Observation space configuration
    obsWidth*: int
    obsHeight*: int
    obsLayers*: int
    
    # Map configuration
    mapWidth*: int
    mapHeight*: int
    numVillages*: int
    
    # Resource configuration
    resourceSpawnRate*: float
    orePerBattery*: int
    batteriesPerHeart*: int
    
    # Combat configuration
    enableCombat*: bool
    clippySpawnRate*: float
    clippyDamage*: int
    
    # Reward configuration
    heartReward*: float
    oreReward*: float
    batteryReward*: float
    woodReward*: float
    waterReward*: float
    wheatReward*: float
    spearReward*: float
    armorReward*: float
    foodReward*: float
    clothReward*: float
    clippyKillReward*: float
    survivalPenalty*: float
    deathPenalty*: float

proc defaultEnvironmentConfig*(): EnvironmentConfig =
  ## Create default environment configuration matching original constants
  EnvironmentConfig(
    # Core game parameters
    numAgents: MapAgents,
    maxSteps: 2000,
    episodeTruncates: false,
    
    # Observation space configuration (match constants)
    obsWidth: ObservationWidth,
    obsHeight: ObservationHeight,
    obsLayers: ObservationLayers,
    
    # Map configuration (match constants)
    mapWidth: MapWidth,
    mapHeight: MapHeight,
    numVillages: MapRoomObjectsHouses,
    
    # Resource configuration
    resourceSpawnRate: 0.1,
    orePerBattery: 3,
    batteriesPerHeart: 2,
    
    # Combat configuration
    enableCombat: true,
    clippySpawnRate: 0.05,
    clippyDamage: 1,
    
    # Reward configuration (match original constants)
    heartReward: 10.0,                  # High reward for completing resource chain
    oreReward: 0.003,                   # RewardMineOre
    batteryReward: 0.01,                # RewardConvertOreToBattery
    woodReward: 0.002,                  # RewardGetWood
    waterReward: 0.001,                 # RewardGetWater
    wheatReward: 0.001,                 # RewardGetWheat
    spearReward: 0.01,                  # RewardCraftSpear
    armorReward: 0.015,                 # RewardCraftArmor
    foodReward: 0.012,                  # RewardCraftFood
    clothReward: 0.012,                 # RewardCraftCloth
    clippyKillReward: 0.1,             # RewardDestroyClippy
    survivalPenalty: -0.01,            # Small per-step penalty
    deathPenalty: -5.0                 # Penalty for agent death
  )