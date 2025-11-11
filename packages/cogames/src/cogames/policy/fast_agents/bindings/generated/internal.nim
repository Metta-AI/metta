when not defined(gcArc) and not defined(gcOrc):
  {.error: "Please use --gc:arc or --gc:orc when using Genny.".}

when (NimMajor, NimMinor, NimPatch) == (1, 6, 2):
  {.error: "Nim 1.6.2 not supported with Genny due to FFI issues.".}
proc fast_agents_init_chook*() {.raises: [], cdecl, exportc, dynlib.} =
  initCHook()

proc fast_agents_random_agent_unref*(x: RandomAgent) {.raises: [], cdecl, exportc, dynlib.} =
  GC_unref(x)

proc fast_agents_new_random_agent*(agent_id: int, environment_config: cstring): RandomAgent {.raises: [], cdecl, exportc, dynlib.} =
  newRandomAgent(agent_id, environment_config.`$`)

proc fast_agents_random_agent_get_agent_id*(random_agent: RandomAgent): int {.raises: [], cdecl, exportc, dynlib.} =
  random_agent.agentId

proc fast_agents_random_agent_set_agent_id*(random_agent: RandomAgent, agentId: int) {.raises: [], cdecl, exportc, dynlib.} =
  random_agent.agentId = agentId

proc fast_agents_random_agent_reset*(agent: RandomAgent) {.raises: [], cdecl, exportc, dynlib.} =
  reset(agent)

proc fast_agents_random_agent_step*(agent: RandomAgent, num_agents: int, num_tokens: int, size_token: int, raw_observations: pointer, num_actions: int, raw_actions: pointer) {.raises: [], cdecl, exportc, dynlib.} =
  step(agent, num_agents, num_tokens, size_token, raw_observations, num_actions, raw_actions)

proc fast_agents_thinky_agent_unref*(x: ThinkyAgent) {.raises: [], cdecl, exportc, dynlib.} =
  GC_unref(x)

proc fast_agents_new_thinky_agent*(agent_id: int, environment_config: cstring): ThinkyAgent {.raises: [], cdecl, exportc, dynlib.} =
  newThinkyAgent(agent_id, environment_config.`$`)

proc fast_agents_thinky_agent_get_agent_id*(thinky_agent: ThinkyAgent): int {.raises: [], cdecl, exportc, dynlib.} =
  thinky_agent.agentId

proc fast_agents_thinky_agent_set_agent_id*(thinky_agent: ThinkyAgent, agentId: int) {.raises: [], cdecl, exportc, dynlib.} =
  thinky_agent.agentId = agentId

proc fast_agents_thinky_agent_reset*(agent: ThinkyAgent) {.raises: [], cdecl, exportc, dynlib.} =
  reset(agent)

proc fast_agents_thinky_agent_step*(agent: ThinkyAgent, num_agents: int, num_tokens: int, size_token: int, raw_observations: pointer, num_actions: int, raw_actions: pointer) {.raises: [], cdecl, exportc, dynlib.} =
  step(agent, num_agents, num_tokens, size_token, raw_observations, num_actions, raw_actions)

proc fast_agents_race_car_agent_unref*(x: RaceCarAgent) {.raises: [], cdecl, exportc, dynlib.} =
  GC_unref(x)

proc fast_agents_new_race_car_agent*(agent_id: int, environment_config: cstring): RaceCarAgent {.raises: [], cdecl, exportc, dynlib.} =
  newRaceCarAgent(agent_id, environment_config.`$`)

proc fast_agents_race_car_agent_get_agent_id*(race_car_agent: RaceCarAgent): int {.raises: [], cdecl, exportc, dynlib.} =
  race_car_agent.agentId

proc fast_agents_race_car_agent_set_agent_id*(race_car_agent: RaceCarAgent, agentId: int) {.raises: [], cdecl, exportc, dynlib.} =
  race_car_agent.agentId = agentId

proc fast_agents_race_car_agent_reset*(agent: RaceCarAgent) {.raises: [], cdecl, exportc, dynlib.} =
  reset(agent)

proc fast_agents_race_car_agent_step*(agent: RaceCarAgent, num_agents: int, num_tokens: int, size_token: int, raw_observations: pointer, num_actions: int, raw_actions: pointer) {.raises: [], cdecl, exportc, dynlib.} =
  step(agent, num_agents, num_tokens, size_token, raw_observations, num_actions, raw_actions)

