when not defined(gcArc) and not defined(gcOrc):
  {.error: "Please use --gc:arc or --gc:orc when using Genny.".}

when (NimMajor, NimMinor, NimPatch) == (1, 6, 2):
  {.error: "Nim 1.6.2 not supported with Genny due to FFI issues.".}
proc heuristic_agents_heuristic_agent_unref*(x: HeuristicAgent) {.raises: [], cdecl, exportc, dynlib.} =
  GC_unref(x)

proc heuristic_agents_new_heuristic_agent*(agent_id: int, environment_config: cstring): HeuristicAgent {.raises: [], cdecl, exportc, dynlib.} =
  newHeuristicAgent(agent_id, environment_config.`$`)

proc heuristic_agents_heuristic_agent_get_agent_id*(heuristic_agent: HeuristicAgent): int {.raises: [], cdecl, exportc, dynlib.} =
  heuristic_agent.agentId

proc heuristic_agents_heuristic_agent_set_agent_id*(heuristic_agent: HeuristicAgent, agentId: int) {.raises: [], cdecl, exportc, dynlib.} =
  heuristic_agent.agentId = agentId

proc heuristic_agents_heuristic_agent_reset*(agent: HeuristicAgent) {.raises: [], cdecl, exportc, dynlib.} =
  reset(agent)

proc heuristic_agents_heuristic_agent_step*(agent: HeuristicAgent, num_agents: int, num_tokens: int, size_token: int, row_observations: pointer, num_actions: int, raw_actions: pointer) {.raises: [], cdecl, exportc, dynlib.} =
  step(agent, num_agents, num_tokens, size_token, row_observations, num_actions, raw_actions)

proc heuristic_agents_init_chook*() {.raises: [], cdecl, exportc, dynlib.} =
  initCHook()

