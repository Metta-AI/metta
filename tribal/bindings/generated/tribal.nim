import bumpy, chroma, unicode, vmath

export bumpy, chroma, unicode, vmath

when defined(windows):
  const libName = "tribal.dll"
elif defined(macosx):
  const libName = "libtribal.dylib"
else:
  const libName = "libtribal.so"

{.push dynlib: libName.}

type TribalError = object of ValueError

const MapAgents* = 15

const ObservationWidth* = 11

const ObservationHeight* = 11

const ObservationLayers* = 19

const MapWidth* = 100

const MapHeight* = 50

type SeqIntObj = object
  reference: pointer

type SeqInt* = ref SeqIntObj

proc tribal_seq_int_unref(x: SeqIntObj) {.importc: "tribal_seq_int_unref", cdecl.}

proc `=destroy`(x: var SeqIntObj) =
  tribal_seq_int_unref(x)

type SeqFloatObj = object
  reference: pointer

type SeqFloat* = ref SeqFloatObj

proc tribal_seq_float_unref(x: SeqFloatObj) {.importc: "tribal_seq_float_unref", cdecl.}

proc `=destroy`(x: var SeqFloatObj) =
  tribal_seq_float_unref(x)

type SeqBoolObj = object
  reference: pointer

type SeqBool* = ref SeqBoolObj

proc tribal_seq_bool_unref(x: SeqBoolObj) {.importc: "tribal_seq_bool_unref", cdecl.}

proc `=destroy`(x: var SeqBoolObj) =
  tribal_seq_bool_unref(x)

type TribalGameConfig* = object
  maxSteps*: int
  orePerBattery*: int
  batteriesPerHeart*: int
  enableCombat*: bool
  clippySpawnRate*: float
  clippyDamage*: int
  heartReward*: float
  oreReward*: float
  batteryReward*: float
  survivalPenalty*: float
  deathPenalty*: float

proc tribalGameConfig*(max_steps: int, ore_per_battery: int, batteries_per_heart: int, enable_combat: bool, clippy_spawn_rate: float, clippy_damage: int, heart_reward: float, ore_reward: float, battery_reward: float, survival_penalty: float, death_penalty: float): TribalGameConfig =
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

type TribalConfig* = object
  game*: TribalGameConfig
  desyncEpisodes*: bool

proc tribalConfig*(game: TribalGameConfig, desync_episodes: bool): TribalConfig =
  result.game = game
  result.desync_episodes = desync_episodes

type TribalEnvObj = object
  reference: pointer

type TribalEnv* = ref TribalEnvObj

proc tribal_tribal_env_unref(x: TribalEnvObj) {.importc: "tribal_tribal_env_unref", cdecl.}

proc `=destroy`(x: var TribalEnvObj) =
  tribal_tribal_env_unref(x)

proc tribal_seq_int_len(s: SeqInt): int {.importc: "tribal_seq_int_len", cdecl.}

proc len*(s: SeqInt): int =
  tribal_seq_int_len(s)

proc tribal_seq_int_add(s: SeqInt, v: int) {.importc: "tribal_seq_int_add", cdecl.}

proc add*(s: SeqInt, v: int) =
  tribal_seq_int_add(s, v)

proc tribal_seq_int_get(s: SeqInt, i: int): int {.importc: "tribal_seq_int_get", cdecl.}

proc `[]`*(s: SeqInt, i: int): int =
  tribal_seq_int_get(s, i)

proc tribal_seq_int_set(s: SeqInt, i: int, v: int) {.importc: "tribal_seq_int_set", cdecl.}

proc `[]=`*(s: SeqInt, i: int, v: int) =
  tribal_seq_int_set(s, i, v)

proc tribal_seq_int_delete(s: SeqInt, i: int) {.importc: "tribal_seq_int_delete", cdecl.}

proc delete*(s: SeqInt, i: int) =
  tribal_seq_int_delete(s, i)

proc tribal_seq_int_clear(s: SeqInt) {.importc: "tribal_seq_int_clear", cdecl.}

proc clear*(s: SeqInt) =
  tribal_seq_int_clear(s)

proc tribal_new_seq_int*(): SeqInt {.importc: "tribal_new_seq_int", cdecl.}

proc newSeqInt*(): SeqInt =
  tribal_new_seq_int()

proc tribal_seq_float_len(s: SeqFloat): int {.importc: "tribal_seq_float_len", cdecl.}

proc len*(s: SeqFloat): int =
  tribal_seq_float_len(s)

proc tribal_seq_float_add(s: SeqFloat, v: float) {.importc: "tribal_seq_float_add", cdecl.}

proc add*(s: SeqFloat, v: float) =
  tribal_seq_float_add(s, v)

proc tribal_seq_float_get(s: SeqFloat, i: int): float {.importc: "tribal_seq_float_get", cdecl.}

proc `[]`*(s: SeqFloat, i: int): float =
  tribal_seq_float_get(s, i)

proc tribal_seq_float_set(s: SeqFloat, i: int, v: float) {.importc: "tribal_seq_float_set", cdecl.}

proc `[]=`*(s: SeqFloat, i: int, v: float) =
  tribal_seq_float_set(s, i, v)

proc tribal_seq_float_delete(s: SeqFloat, i: int) {.importc: "tribal_seq_float_delete", cdecl.}

proc delete*(s: SeqFloat, i: int) =
  tribal_seq_float_delete(s, i)

proc tribal_seq_float_clear(s: SeqFloat) {.importc: "tribal_seq_float_clear", cdecl.}

proc clear*(s: SeqFloat) =
  tribal_seq_float_clear(s)

proc tribal_new_seq_float*(): SeqFloat {.importc: "tribal_new_seq_float", cdecl.}

proc newSeqFloat*(): SeqFloat =
  tribal_new_seq_float()

proc tribal_seq_bool_len(s: SeqBool): int {.importc: "tribal_seq_bool_len", cdecl.}

proc len*(s: SeqBool): int =
  tribal_seq_bool_len(s)

proc tribal_seq_bool_add(s: SeqBool, v: bool) {.importc: "tribal_seq_bool_add", cdecl.}

proc add*(s: SeqBool, v: bool) =
  tribal_seq_bool_add(s, v)

proc tribal_seq_bool_get(s: SeqBool, i: int): bool {.importc: "tribal_seq_bool_get", cdecl.}

proc `[]`*(s: SeqBool, i: int): bool =
  tribal_seq_bool_get(s, i)

proc tribal_seq_bool_set(s: SeqBool, i: int, v: bool) {.importc: "tribal_seq_bool_set", cdecl.}

proc `[]=`*(s: SeqBool, i: int, v: bool) =
  tribal_seq_bool_set(s, i, v)

proc tribal_seq_bool_delete(s: SeqBool, i: int) {.importc: "tribal_seq_bool_delete", cdecl.}

proc delete*(s: SeqBool, i: int) =
  tribal_seq_bool_delete(s, i)

proc tribal_seq_bool_clear(s: SeqBool) {.importc: "tribal_seq_bool_clear", cdecl.}

proc clear*(s: SeqBool) =
  tribal_seq_bool_clear(s)

proc tribal_new_seq_bool*(): SeqBool {.importc: "tribal_new_seq_bool", cdecl.}

proc newSeqBool*(): SeqBool =
  tribal_new_seq_bool()

proc tribal_check_error(): bool {.importc: "tribal_check_error", cdecl.}

proc checkError*(): bool {.inline.} =
  result = tribal_check_error()

proc tribal_take_error(): cstring {.importc: "tribal_take_error", cdecl.}

proc takeError*(): cstring {.inline.} =
  result = tribal_take_error()

proc tribal_new_tribal_env(config: TribalConfig): TribalEnv {.importc: "tribal_new_tribal_env", cdecl.}

proc newTribalEnv*(config: TribalConfig): TribalEnv {.inline.} =
  result = tribal_new_tribal_env(config)

proc tribal_tribal_env_reset_env(tribal: TribalEnv) {.importc: "tribal_tribal_env_reset_env", cdecl.}

proc resetEnv*(tribal: TribalEnv) {.inline.} =
  tribal_tribal_env_reset_env(tribal)

proc tribal_tribal_env_step(tribal: TribalEnv, actions: SeqInt): bool {.importc: "tribal_tribal_env_step", cdecl.}

proc step*(tribal: TribalEnv, actions: SeqInt): bool {.inline.} =
  result = tribal_tribal_env_step(tribal, actions)

proc tribal_tribal_env_get_observations(tribal: TribalEnv): SeqInt {.importc: "tribal_tribal_env_get_observations", cdecl.}

proc getObservations*(tribal: TribalEnv): SeqInt {.inline.} =
  result = tribal_tribal_env_get_observations(tribal)

proc tribal_tribal_env_get_rewards(tribal: TribalEnv): SeqFloat {.importc: "tribal_tribal_env_get_rewards", cdecl.}

proc getRewards*(tribal: TribalEnv): SeqFloat {.inline.} =
  result = tribal_tribal_env_get_rewards(tribal)

proc tribal_tribal_env_get_terminated(tribal: TribalEnv): SeqBool {.importc: "tribal_tribal_env_get_terminated", cdecl.}

proc getTerminated*(tribal: TribalEnv): SeqBool {.inline.} =
  result = tribal_tribal_env_get_terminated(tribal)

proc tribal_tribal_env_get_truncated(tribal: TribalEnv): SeqBool {.importc: "tribal_tribal_env_get_truncated", cdecl.}

proc getTruncated*(tribal: TribalEnv): SeqBool {.inline.} =
  result = tribal_tribal_env_get_truncated(tribal)

proc tribal_tribal_env_get_current_step(tribal: TribalEnv): int {.importc: "tribal_tribal_env_get_current_step", cdecl.}

proc getCurrentStep*(tribal: TribalEnv): int {.inline.} =
  result = tribal_tribal_env_get_current_step(tribal)

proc tribal_tribal_env_is_episode_done(tribal: TribalEnv): bool {.importc: "tribal_tribal_env_is_episode_done", cdecl.}

proc isEpisodeDone*(tribal: TribalEnv): bool {.inline.} =
  result = tribal_tribal_env_is_episode_done(tribal)

proc tribal_tribal_env_render_text(tribal: TribalEnv): cstring {.importc: "tribal_tribal_env_render_text", cdecl.}

proc renderText*(tribal: TribalEnv): cstring {.inline.} =
  result = tribal_tribal_env_render_text(tribal)

proc tribal_default_max_steps(): int {.importc: "tribal_default_max_steps", cdecl.}

proc defaultMaxSteps*(): int {.inline.} =
  result = tribal_default_max_steps()

proc tribal_default_tribal_config(): TribalConfig {.importc: "tribal_default_tribal_config", cdecl.}

proc defaultTribalConfig*(): TribalConfig {.inline.} =
  result = tribal_default_tribal_config()

