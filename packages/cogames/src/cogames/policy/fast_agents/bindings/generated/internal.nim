when not defined(gcArc) and not defined(gcOrc):
  {.error: "Please use --gc:arc or --gc:orc when using Genny.".}

when (NimMajor, NimMinor, NimPatch) == (1, 6, 2):
  {.error: "Nim 1.6.2 not supported with Genny due to FFI issues.".}
proc fast_agents_fast_agents_unref*(x: FastAgents) {.raises: [], cdecl, exportc, dynlib.} =
  GC_unref(x)

proc fast_agents_new_fast_agents*(agent_id: int, environment_config: cstring): FastAgents {.raises: [], cdecl, exportc, dynlib.} =
  newFastAgents(agent_id, environment_config.`$`)

proc fast_agents_fast_agents_get_agent_id*(fast_agents: FastAgents): int {.raises: [], cdecl, exportc, dynlib.} =
  fast_agents.agentId

proc fast_agents_fast_agents_set_agent_id*(fast_agents: FastAgents, agentId: int) {.raises: [], cdecl, exportc, dynlib.} =
  fast_agents.agentId = agentId

proc fast_agents_fast_agents_reset*(agent: FastAgents) {.raises: [], cdecl, exportc, dynlib.} =
  reset(agent)

proc fast_agents_fast_agents_step*(agent: FastAgents, num_agents: int, num_tokens: int, size_token: int, raw_observations: pointer, num_actions: int, raw_actions: pointer) {.raises: [], cdecl, exportc, dynlib.} =
  step(agent, num_agents, num_tokens, size_token, raw_observations, num_actions, raw_actions)

proc fast_agents_init_chook*() {.raises: [], cdecl, exportc, dynlib.} =
  initCHook()

