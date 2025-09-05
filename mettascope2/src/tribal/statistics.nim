## Statistics tracking for the environment
## Handles action counting and episode statistics reporting

import std/strformat
import environment_core

proc getEpisodeStats*(env: Environment): string =
  ## Get the episode statistics as a formatted string
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
  display "action.move", actionMove
  display "action.noop", actionNoop
  display "action.rotate", actionRotate
  display "action.swap", actionSwap
  display "action.use", actionUse
  display "action.use.altar", actionUseAltar
  display "action.use.converter", actionUseConverter
  display "action.use.mine", actionUseMine
  display "action.get", actionGet
  display "action.get.water", actionGetWater
  display "action.get.wheat", actionGetWheat
  display "action.get.wood", actionGetWood
  
  return result

proc resetStats*(env: Environment) =
  ## Reset all statistics
  env.stats.setLen(0)
  for i in 0 ..< MapAgents:
    env.stats.add(Stats())

proc initStats*(env: Environment) =
  ## Initialize statistics for all agents
  resetStats(env)