## Test configuration integration between Nim and Python layers
## Verifies that configuration defaults flow correctly from Nim to Python

import unittest
import ../src/tribal/environment
import ../bindings/tribal_bindings

suite "Configuration Integration Tests":
  
  test "Nim environment defaults match binding defaults":
    # Test that the binding defaults match the core environment defaults
    let envDefaults = defaultEnvironmentConfig()
    let bindingDefaults = defaultTribalConfig()
    
    # Check key values match between layers
    check envDefaults.maxSteps == bindingDefaults.game.maxSteps
    check envDefaults.orePerBattery == bindingDefaults.game.orePerBattery
    check envDefaults.batteriesPerHeart == bindingDefaults.game.batteriesPerHeart
    check envDefaults.enableCombat == bindingDefaults.game.enableCombat
    check envDefaults.clippySpawnRate == bindingDefaults.game.clippySpawnRate
    check envDefaults.clippyDamage == bindingDefaults.game.clippyDamage
    check envDefaults.heartReward == bindingDefaults.game.heartReward
    check envDefaults.oreReward == bindingDefaults.game.oreReward
    check envDefaults.batteryReward == bindingDefaults.game.batteryReward
    check envDefaults.survivalPenalty == bindingDefaults.game.survivalPenalty
    check envDefaults.deathPenalty == bindingDefaults.game.deathPenalty

  test "Environment creation uses correct defaults":
    # Test that environment creation respects default configuration
    let config = defaultEnvironmentConfig()
    let env = newEnvironment(config)
    
    check env.config.maxSteps == config.maxSteps
    check env.config.enableCombat == config.enableCombat
    check env.config.heartReward == config.heartReward
    check env.config.batteryReward == config.batteryReward

  test "Configuration validation":
    # Test that configuration values are within expected ranges
    let config = defaultEnvironmentConfig()
    
    # Basic sanity checks
    check config.maxSteps > 0
    check config.orePerBattery > 0
    check config.batteriesPerHeart > 0
    check config.clippySpawnRate >= 0.0 and config.clippySpawnRate <= 1.0
    check config.clippyDamage >= 0
    
    # Reward values should be reasonable
    check config.heartReward > 0.0  # Positive reward for progress
    check config.survivalPenalty <= 0.0  # Should be penalty or neutral
    check config.deathPenalty <= 0.0  # Should be penalty

when isMainModule:
  discard  # The unittest framework automatically runs tests when compiled and run