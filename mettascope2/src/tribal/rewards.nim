## Shaped Rewards Module
## Defines reward values for different actions to encourage meaningful agent behavior
## These rewards guide agents through: exploration → resource gathering → crafting → combat → cooperation

# ============ Resource Gathering Rewards ============
# Small rewards for collecting basic resources
const
  RewardGetWater* = 0.001      # Collecting water from tiles
  RewardGetWheat* = 0.001      # Harvesting wheat 
  RewardGetWood* = 0.002       # Chopping wood (slightly higher as it's needed for spears)
  RewardMineOre* = 0.003       # Mining ore (first step in battery chain)

# ============ Crafting & Production Rewards ============
# Medium rewards for transforming resources
const
  RewardConvertOreToBattery* = 0.01   # Using converter to make batteries
  RewardCraftSpear* = 0.01            # Using forge to craft spear
  RewardCraftArmor* = 0.015           # Using armory to craft armor  
  RewardCraftFood* = 0.012            # Using clay oven to craft food
  RewardCraftCloth* = 0.012           # Using weaving loom to craft cloth

# ============ Combat & Defense Rewards ============
# High rewards for successful combat
const
  RewardDestroyClippy* = 0.1          # Destroying a clippy with spear

# ============ Cooperation & Contribution Rewards ============
# Highest rewards for team contributions (handled in main code as 1.0)
# RewardDepositBatteryAtAltar = 1.0  # Contributing battery to altar (main goal)