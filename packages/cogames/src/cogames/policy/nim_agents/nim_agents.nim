import
  genny, fidget2/measure,
  random_agents, thinky_agents, racecar_agents, ladybug_agent


proc ctrlCHandler() {.noconv.} =
  echo "\nNim DLL caught ctrl-c, exiting..."
  quit(0)

proc nim_agents_init_chook*() =
  setControlCHook(ctrlCHandler)
  echo "NimAgents initialized"

proc startMeasure() =
  startTrace()

proc endMeasure() =
  try:
    endTrace()
    dumpMeasures(0.0, "tmp/trace.json")
  except:
    echo "Error ending measure: ", getCurrentExceptionMsg()

exportProcs:
  nim_agents_init_chook
  startMeasure
  endMeasure

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

exportRefObject LadybugPolicy:
  constructor:
    newLadybugPolicy(string)
  procs:
    stepBatch(LadybugPolicy, pointer, int, int, int, int, pointer, int, pointer)

writeFiles("bindings/generated", "NimAgents")

include bindings/generated/internal
