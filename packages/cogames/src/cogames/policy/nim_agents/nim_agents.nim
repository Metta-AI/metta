import
  genny,
  random_agents, thinky_agents, race_car_agents

proc ctrlCHandler() {.noconv.} =
  echo "\nNim DLL caught ctrl-c, exiting..."
  quit(0)

proc nim_agents_init_chook*() =
  setControlCHook(ctrlCHandler)
  echo "NimAgents initialized"


exportProcs:
  nim_agents_init_chook

exportRefObject RandomPolicy:
  constructor:
    newRandomPolicy(string)
  procs:
    stepBatch(RandomPolicy, pointer, int, int, int, int, pointer, int, pointer)

exportRefObject ThinkyPolicy:
  constructor:
    newThinkyPolicy(string)
  procs:
    stepBatch(ThinkyPolicy, pointer, int, int, int, int, pointer, int, pointer)

exportRefObject RaceCarPolicy:
  constructor:
    newRaceCarPolicy(string)
  procs:
    stepBatch(RaceCarPolicy, pointer, int, int, int, int, pointer, int, pointer)

writeFiles("bindings/generated", "NimAgents")

include bindings/generated/internal
