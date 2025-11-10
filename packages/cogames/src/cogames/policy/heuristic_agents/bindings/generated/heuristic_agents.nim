import bumpy, chroma, unicode, vmath

export bumpy, chroma, unicode, vmath

when defined(windows):
  const libName = "heuristic_agents.dll"
elif defined(macosx):
  const libName = "libheuristic_agents.dylib"
else:
  const libName = "libheuristic_agents.so"

{.push dynlib: libName.}

type HeuristicAgentsError = object of ValueError

type HeuristicAgentObj = object
  reference: pointer

type HeuristicAgent* = ref HeuristicAgentObj

proc heuristic_agents_heuristic_agent_unref(x: HeuristicAgentObj) {.importc: "heuristic_agents_heuristic_agent_unref", cdecl.}

proc `=destroy`(x: var HeuristicAgentObj) =
  heuristic_agents_heuristic_agent_unref(x)

proc heuristic_agents_new_heuristic_agent(agent_id: int, environment_config: cstring): HeuristicAgent {.importc: "heuristic_agents_new_heuristic_agent", cdecl.}

proc newHeuristicAgent*(agentId: int, environmentConfig: string): HeuristicAgent {.inline.} =
  result = heuristic_agents_new_heuristic_agent(agentId, environmentConfig.cstring)

proc heuristic_agents_heuristic_agent_get_agent_id(heuristicAgent: HeuristicAgent): int {.importc: "heuristic_agents_heuristic_agent_get_agent_id", cdecl.}

proc agentId*(heuristicAgent: HeuristicAgent): int {.inline.} =
  heuristic_agents_heuristic_agent_get_agent_id(heuristicAgent)

proc heuristic_agents_heuristic_agent_set_agent_id(heuristicAgent: HeuristicAgent, agentId: int) {.importc: "heuristic_agents_heuristic_agent_set_agent_id", cdecl.}

proc `agentId=`*(heuristicAgent: HeuristicAgent, agentId: int) =
  heuristic_agents_heuristic_agent_set_agent_id(heuristicAgent, agentId)

proc heuristic_agents_heuristic_agent_reset(agent: HeuristicAgent) {.importc: "heuristic_agents_heuristic_agent_reset", cdecl.}

proc reset*(agent: HeuristicAgent) {.inline.} =
  heuristic_agents_heuristic_agent_reset(agent)

proc heuristic_agents_heuristic_agent_step(agent: HeuristicAgent, num_agents: int, num_tokens: int, size_token: int, row_observations: pointer, num_actions: int, raw_actions: pointer) {.importc: "heuristic_agents_heuristic_agent_step", cdecl.}

proc step*(agent: HeuristicAgent, numAgents: int, numTokens: int, sizeToken: int, rowObservations: pointer, numActions: int, rawActions: pointer) {.inline.} =
  heuristic_agents_heuristic_agent_step(agent, numAgents, numTokens, sizeToken, rowObservations, numActions, rawActions)

proc heuristic_agents_init_chook() {.importc: "heuristic_agents_init_chook", cdecl.}

proc initCHook*() {.inline.} =
  heuristic_agents_init_chook()

