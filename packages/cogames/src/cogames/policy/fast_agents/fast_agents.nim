import
  std/[strformat, strutils, tables, random, sets],
  genny, jsony,
  common, random_agents, thinky_agents, race_car_agents

exportProcs:
  initCHook

exportRefObject RandomAgent:
  constructor:
    newRandomAgent(int, string)
  fields:
    agentId
  procs:
    reset(RandomAgent)
    step(RandomAgent, int, int, int, pointer, int, pointer)

exportRefObject ThinkyAgent:
  constructor:
    newThinkyAgent(int, string)
  fields:
    agentId
  procs:
    reset(ThinkyAgent)
    step(ThinkyAgent, int, int, int, pointer, int, pointer)

exportRefObject RaceCarAgent:
  constructor:
    newRaceCarAgent(int, string)
  fields:
    agentId
  procs:
    reset(RaceCarAgent)
    step(RaceCarAgent, int, int, int, pointer, int, pointer)

writeFiles("bindings/generated", "FastAgents")

include bindings/generated/internal
