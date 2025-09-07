when not defined(gcArc) and not defined(gcOrc):
  {.error: "Please use --gc:arc or --gc:orc when using Genny.".}

when (NimMajor, NimMinor, NimPatch) == (1, 6, 2):
  {.error: "Nim 1.6.2 not supported with Genny due to FFI issues.".}
type SeqInt* = ref object
  s: seq[int]

proc tribal_new_seq_int*(): SeqInt {.raises: [], cdecl, exportc, dynlib.} =
  SeqInt()

proc tribal_seq_int_len*(s: SeqInt): int {.raises: [], cdecl, exportc, dynlib.} =
  s.s.len

proc tribal_seq_int_add*(s: SeqInt, v: int) {.raises: [], cdecl, exportc, dynlib.} =
  s.s.add(v)

proc tribal_seq_int_get*(s: SeqInt, i: int): int {.raises: [], cdecl, exportc, dynlib.} =
  s.s[i]

proc tribal_seq_int_set*(s: SeqInt, i: int, v: int) {.raises: [], cdecl, exportc, dynlib.} =
  s.s[i] = v

proc tribal_seq_int_delete*(s: SeqInt, i: int) {.raises: [], cdecl, exportc, dynlib.} =
  s.s.delete(i)

proc tribal_seq_int_clear*(s: SeqInt) {.raises: [], cdecl, exportc, dynlib.} =
  s.s.setLen(0)

proc tribal_seq_int_unref*(s: SeqInt) {.raises: [], cdecl, exportc, dynlib.} =
  GC_unref(s)

type SeqFloat* = ref object
  s: seq[float]

proc tribal_new_seq_float*(): SeqFloat {.raises: [], cdecl, exportc, dynlib.} =
  SeqFloat()

proc tribal_seq_float_len*(s: SeqFloat): int {.raises: [], cdecl, exportc, dynlib.} =
  s.s.len

proc tribal_seq_float_add*(s: SeqFloat, v: float) {.raises: [], cdecl, exportc, dynlib.} =
  s.s.add(v)

proc tribal_seq_float_get*(s: SeqFloat, i: int): float {.raises: [], cdecl, exportc, dynlib.} =
  s.s[i]

proc tribal_seq_float_set*(s: SeqFloat, i: int, v: float) {.raises: [], cdecl, exportc, dynlib.} =
  s.s[i] = v

proc tribal_seq_float_delete*(s: SeqFloat, i: int) {.raises: [], cdecl, exportc, dynlib.} =
  s.s.delete(i)

proc tribal_seq_float_clear*(s: SeqFloat) {.raises: [], cdecl, exportc, dynlib.} =
  s.s.setLen(0)

proc tribal_seq_float_unref*(s: SeqFloat) {.raises: [], cdecl, exportc, dynlib.} =
  GC_unref(s)

type SeqBool* = ref object
  s: seq[bool]

proc tribal_new_seq_bool*(): SeqBool {.raises: [], cdecl, exportc, dynlib.} =
  SeqBool()

proc tribal_seq_bool_len*(s: SeqBool): int {.raises: [], cdecl, exportc, dynlib.} =
  s.s.len

proc tribal_seq_bool_add*(s: SeqBool, v: bool) {.raises: [], cdecl, exportc, dynlib.} =
  s.s.add(v)

proc tribal_seq_bool_get*(s: SeqBool, i: int): bool {.raises: [], cdecl, exportc, dynlib.} =
  s.s[i]

proc tribal_seq_bool_set*(s: SeqBool, i: int, v: bool) {.raises: [], cdecl, exportc, dynlib.} =
  s.s[i] = v

proc tribal_seq_bool_delete*(s: SeqBool, i: int) {.raises: [], cdecl, exportc, dynlib.} =
  s.s.delete(i)

proc tribal_seq_bool_clear*(s: SeqBool) {.raises: [], cdecl, exportc, dynlib.} =
  s.s.setLen(0)

proc tribal_seq_bool_unref*(s: SeqBool) {.raises: [], cdecl, exportc, dynlib.} =
  GC_unref(s)

proc tribal_check_error*(): bool {.raises: [], cdecl, exportc, dynlib.} =
  checkError()

proc tribal_take_error*(): cstring {.raises: [], cdecl, exportc, dynlib.} =
  takeError().cstring

proc tribal_tribal_game_config*(max_steps: int, ore_per_battery: int, batteries_per_heart: int, enable_combat: bool, clippy_spawn_rate: float, clippy_damage: int, heart_reward: float, ore_reward: float, battery_reward: float, survival_penalty: float, death_penalty: float): TribalGameConfig {.raises: [], cdecl, exportc, dynlib.} =
  result.max_steps = max_steps
  result.ore_per_battery = ore_per_battery
  result.batteries_per_heart = batteries_per_heart
  result.enable_combat = enable_combat
  result.clippy_spawn_rate = clippy_spawn_rate
  result.clippy_damage = clippy_damage
  result.heart_reward = heart_reward
  result.ore_reward = ore_reward
  result.battery_reward = battery_reward
  result.survival_penalty = survival_penalty
  result.death_penalty = death_penalty

proc tribal_tribal_game_config_eq*(a, b: TribalGameConfig): bool {.raises: [], cdecl, exportc, dynlib.}=
  a.max_steps == b.max_steps and a.ore_per_battery == b.ore_per_battery and a.batteries_per_heart == b.batteries_per_heart and a.enable_combat == b.enable_combat and a.clippy_spawn_rate == b.clippy_spawn_rate and a.clippy_damage == b.clippy_damage and a.heart_reward == b.heart_reward and a.ore_reward == b.ore_reward and a.battery_reward == b.battery_reward and a.survival_penalty == b.survival_penalty and a.death_penalty == b.death_penalty

proc tribal_tribal_config*(game: TribalGameConfig, desync_episodes: bool): TribalConfig {.raises: [], cdecl, exportc, dynlib.} =
  result.game = game
  result.desync_episodes = desync_episodes

proc tribal_tribal_config_eq*(a, b: TribalConfig): bool {.raises: [], cdecl, exportc, dynlib.}=
  a.game == b.game and a.desync_episodes == b.desync_episodes

proc tribal_tribal_env_unref*(x: TribalEnv) {.raises: [], cdecl, exportc, dynlib.} =
  GC_unref(x)

proc tribal_new_tribal_env*(config: TribalConfig): TribalEnv {.raises: [], cdecl, exportc, dynlib.} =
  newTribalEnv(config)

proc tribal_tribal_env_reset_env*(tribal: TribalEnv) {.raises: [], cdecl, exportc, dynlib.} =
  resetEnv(tribal)

proc tribal_tribal_env_step*(tribal: TribalEnv, actions: SeqInt): bool {.raises: [], cdecl, exportc, dynlib.} =
  step(tribal, actions.s)

proc tribal_tribal_env_get_observations*(tribal: TribalEnv): SeqInt {.raises: [], cdecl, exportc, dynlib.} =
  SeqInt(s: getObservations(tribal))

proc tribal_tribal_env_get_rewards*(tribal: TribalEnv): SeqFloat {.raises: [], cdecl, exportc, dynlib.} =
  SeqFloat(s: getRewards(tribal))

proc tribal_tribal_env_get_terminated*(tribal: TribalEnv): SeqBool {.raises: [], cdecl, exportc, dynlib.} =
  SeqBool(s: getTerminated(tribal))

proc tribal_tribal_env_get_truncated*(tribal: TribalEnv): SeqBool {.raises: [], cdecl, exportc, dynlib.} =
  SeqBool(s: getTruncated(tribal))

proc tribal_tribal_env_get_current_step*(tribal: TribalEnv): int {.raises: [], cdecl, exportc, dynlib.} =
  getCurrentStep(tribal)

proc tribal_tribal_env_is_episode_done*(tribal: TribalEnv): bool {.raises: [], cdecl, exportc, dynlib.} =
  isEpisodeDone(tribal)

proc tribal_tribal_env_render_text*(tribal: TribalEnv): cstring {.raises: [], cdecl, exportc, dynlib.} =
  renderText(tribal).cstring

proc tribal_default_max_steps*(): int {.raises: [], cdecl, exportc, dynlib.} =
  defaultMaxSteps()

proc tribal_default_tribal_config*(): TribalConfig {.raises: [], cdecl, exportc, dynlib.} =
  defaultTribalConfig()

