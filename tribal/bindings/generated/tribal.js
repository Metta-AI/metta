var ffi = require('ffi-napi');
var Struct = require("ref-struct-napi");
var ArrayType = require('ref-array-napi');

var dll = {};

function TribalException(message) {
  this.message = message;
  this.name = 'TribalException';
}

SeqInt = Struct({'nimRef': 'uint64'});
SeqInt.prototype.isNull = function(){
  return this.nimRef == 0;
};
SeqInt.prototype.isEqual = function(other){
  return this.nimRef == other.nimRef;
};
SeqInt.prototype.unref = function(){
  return dll.tribal_seq_int_unref(this)
};
function seqInt(){
  return dll.tribal_new_seq_int();
}
SeqInt.prototype.length = function(){
  return dll.tribal_seq_int_len(this)
};
SeqInt.prototype.get = function(index){
  return dll.tribal_seq_int_get(this, index)
};
SeqInt.prototype.set = function(index, value){
  dll.tribal_seq_int_set(this, index, value)
};
SeqInt.prototype.delete = function(index){
  dll.tribal_seq_int_delete(this, index)
};
SeqInt.prototype.add = function(value){
  dll.tribal_seq_int_add(this, value)
};
SeqInt.prototype.clear = function(){
  dll.tribal_seq_int_clear(this)
};
SeqFloat = Struct({'nimRef': 'uint64'});
SeqFloat.prototype.isNull = function(){
  return this.nimRef == 0;
};
SeqFloat.prototype.isEqual = function(other){
  return this.nimRef == other.nimRef;
};
SeqFloat.prototype.unref = function(){
  return dll.tribal_seq_float_unref(this)
};
function seqFloat(){
  return dll.tribal_new_seq_float();
}
SeqFloat.prototype.length = function(){
  return dll.tribal_seq_float_len(this)
};
SeqFloat.prototype.get = function(index){
  return dll.tribal_seq_float_get(this, index)
};
SeqFloat.prototype.set = function(index, value){
  dll.tribal_seq_float_set(this, index, value)
};
SeqFloat.prototype.delete = function(index){
  dll.tribal_seq_float_delete(this, index)
};
SeqFloat.prototype.add = function(value){
  dll.tribal_seq_float_add(this, value)
};
SeqFloat.prototype.clear = function(){
  dll.tribal_seq_float_clear(this)
};
SeqBool = Struct({'nimRef': 'uint64'});
SeqBool.prototype.isNull = function(){
  return this.nimRef == 0;
};
SeqBool.prototype.isEqual = function(other){
  return this.nimRef == other.nimRef;
};
SeqBool.prototype.unref = function(){
  return dll.tribal_seq_bool_unref(this)
};
function seqBool(){
  return dll.tribal_new_seq_bool();
}
SeqBool.prototype.length = function(){
  return dll.tribal_seq_bool_len(this)
};
SeqBool.prototype.get = function(index){
  return dll.tribal_seq_bool_get(this, index)
};
SeqBool.prototype.set = function(index, value){
  dll.tribal_seq_bool_set(this, index, value)
};
SeqBool.prototype.delete = function(index){
  dll.tribal_seq_bool_delete(this, index)
};
SeqBool.prototype.add = function(value){
  dll.tribal_seq_bool_add(this, value)
};
SeqBool.prototype.clear = function(){
  dll.tribal_seq_bool_clear(this)
};
function checkError(){
  result = dll.tribal_check_error()
  return result
}

function takeError(){
  result = dll.tribal_take_error()
  return result
}

const TribalGameConfig = Struct({
  'maxSteps':'int64',
  'orePerBattery':'int64',
  'batteriesPerHeart':'int64',
  'enableCombat':'bool',
  'clippySpawnRate':'double',
  'clippyDamage':'int64',
  'heartReward':'double',
  'oreReward':'double',
  'batteryReward':'double',
  'survivalPenalty':'double',
  'deathPenalty':'double'
})
tribalGameConfig = function(max_steps, ore_per_battery, batteries_per_heart, enable_combat, clippy_spawn_rate, clippy_damage, heart_reward, ore_reward, battery_reward, survival_penalty, death_penalty){
  var v = new TribalGameConfig();
  v.max_steps = max_steps
  v.ore_per_battery = ore_per_battery
  v.batteries_per_heart = batteries_per_heart
  v.enable_combat = enable_combat
  v.clippy_spawn_rate = clippy_spawn_rate
  v.clippy_damage = clippy_damage
  v.heart_reward = heart_reward
  v.ore_reward = ore_reward
  v.battery_reward = battery_reward
  v.survival_penalty = survival_penalty
  v.death_penalty = death_penalty
  return v;
}
TribalGameConfig.prototype.isEqual = function(other){
  return self.maxSteps == other.maxSteps && self.orePerBattery == other.orePerBattery && self.batteriesPerHeart == other.batteriesPerHeart && self.enableCombat == other.enableCombat && self.clippySpawnRate == other.clippySpawnRate && self.clippyDamage == other.clippyDamage && self.heartReward == other.heartReward && self.oreReward == other.oreReward && self.batteryReward == other.batteryReward && self.survivalPenalty == other.survivalPenalty && self.deathPenalty == other.deathPenalty;
};

const TribalConfig = Struct({
  'game':TribalGameConfig,
  'desyncEpisodes':'bool'
})
tribalConfig = function(game, desync_episodes){
  var v = new TribalConfig();
  v.game = game
  v.desync_episodes = desync_episodes
  return v;
}
TribalConfig.prototype.isEqual = function(other){
  return self.game == other.game && self.desyncEpisodes == other.desyncEpisodes;
};

TribalEnv = Struct({'nimRef': 'uint64'});
TribalEnv.prototype.isNull = function(){
  return this.nimRef == 0;
};
TribalEnv.prototype.isEqual = function(other){
  return this.nimRef == other.nimRef;
};
TribalEnv.prototype.unref = function(){
  return dll.tribal_tribal_env_unref(this)
};
function newTribalEnv(config){
  var result = dll.tribal_new_tribal_env(config)
  const registry = new FinalizationRegistry(function(obj) {
    console.log("js unref")
    obj.unref()
  });
  registry.register(result, null);
  return result
}

/**
 * Reset the environment to initial state
 */
TribalEnv.prototype.resetEnv = function(){
  dll.tribal_tribal_env_reset_env(this)
}

/**
 * Step environment with actions
 * actions: flat sequence of [action_type, argument] pairs
 * Length should be MapAgents * 2
 */
TribalEnv.prototype.step = function(actions){
  result = dll.tribal_tribal_env_step(this, actions)
  return result
}

/**
 * Get current observations as flat sequence
 */
TribalEnv.prototype.getObservations = function(){
  result = dll.tribal_tribal_env_get_observations(this)
  return result
}

/**
 * Get current step rewards for each agent
 */
TribalEnv.prototype.getRewards = function(){
  result = dll.tribal_tribal_env_get_rewards(this)
  return result
}

/**
 * Get terminated status for each agent
 */
TribalEnv.prototype.getTerminated = function(){
  result = dll.tribal_tribal_env_get_terminated(this)
  return result
}

/**
 * Get truncated status for each agent  
 */
TribalEnv.prototype.getTruncated = function(){
  result = dll.tribal_tribal_env_get_truncated(this)
  return result
}

/**
 * Get current step number
 */
TribalEnv.prototype.getCurrentStep = function(){
  result = dll.tribal_tribal_env_get_current_step(this)
  return result
}

/**
 * Check if episode should end
 */
TribalEnv.prototype.isEpisodeDone = function(){
  result = dll.tribal_tribal_env_is_episode_done(this)
  return result
}

/**
 * Get text rendering of current state
 */
TribalEnv.prototype.renderText = function(){
  result = dll.tribal_tribal_env_render_text(this)
  return result
}

/**
 * Get default max steps value
 */
function defaultMaxSteps(){
  result = dll.tribal_default_max_steps()
  return result
}

/**
 * Create default tribal configuration
 */
function defaultTribalConfig(){
  result = dll.tribal_default_tribal_config()
  return result
}


var dllPath = ""
if(process.platform == "win32") {
  dllPath = __dirname + '/tribal.dll'
} else if (process.platform == "darwin") {
  dllPath = __dirname + '/libtribal.dylib'
} else {
  dllPath = __dirname + '/libtribal.so'
}

dll = ffi.Library(dllPath, {
  'tribal_seq_int_unref': ['void', [SeqInt]],
  'tribal_new_seq_int': [SeqInt, []],
  'tribal_seq_int_len': ['uint64', [SeqInt]],
  'tribal_seq_int_get': ['int64', [SeqInt, 'uint64']],
  'tribal_seq_int_set': ['void', [SeqInt, 'uint64', 'int64']],
  'tribal_seq_int_delete': ['void', [SeqInt, 'uint64']],
  'tribal_seq_int_add': ['void', [SeqInt, 'int64']],
  'tribal_seq_int_clear': ['void', [SeqInt]],
  'tribal_seq_float_unref': ['void', [SeqFloat]],
  'tribal_new_seq_float': [SeqFloat, []],
  'tribal_seq_float_len': ['uint64', [SeqFloat]],
  'tribal_seq_float_get': ['double', [SeqFloat, 'uint64']],
  'tribal_seq_float_set': ['void', [SeqFloat, 'uint64', 'double']],
  'tribal_seq_float_delete': ['void', [SeqFloat, 'uint64']],
  'tribal_seq_float_add': ['void', [SeqFloat, 'double']],
  'tribal_seq_float_clear': ['void', [SeqFloat]],
  'tribal_seq_bool_unref': ['void', [SeqBool]],
  'tribal_new_seq_bool': [SeqBool, []],
  'tribal_seq_bool_len': ['uint64', [SeqBool]],
  'tribal_seq_bool_get': ['bool', [SeqBool, 'uint64']],
  'tribal_seq_bool_set': ['void', [SeqBool, 'uint64', 'bool']],
  'tribal_seq_bool_delete': ['void', [SeqBool, 'uint64']],
  'tribal_seq_bool_add': ['void', [SeqBool, 'bool']],
  'tribal_seq_bool_clear': ['void', [SeqBool]],
  'tribal_check_error': ['bool', []],
  'tribal_take_error': ['string', []],
  'tribal_tribal_env_unref': ['void', [TribalEnv]],
  'tribal_new_tribal_env': [TribalEnv, [TribalConfig]],
  'tribal_tribal_env_reset_env': ['void', [TribalEnv]],
  'tribal_tribal_env_step': ['bool', [TribalEnv, SeqInt]],
  'tribal_tribal_env_get_observations': [SeqInt, [TribalEnv]],
  'tribal_tribal_env_get_rewards': [SeqFloat, [TribalEnv]],
  'tribal_tribal_env_get_terminated': [SeqBool, [TribalEnv]],
  'tribal_tribal_env_get_truncated': [SeqBool, [TribalEnv]],
  'tribal_tribal_env_get_current_step': ['int64', [TribalEnv]],
  'tribal_tribal_env_is_episode_done': ['bool', [TribalEnv]],
  'tribal_tribal_env_render_text': ['string', [TribalEnv]],
  'tribal_default_max_steps': ['int64', []],
  'tribal_default_tribal_config': [TribalConfig, []],
});

exports.MAP_AGENTS = 15
exports.OBSERVATION_WIDTH = 11
exports.OBSERVATION_HEIGHT = 11
exports.OBSERVATION_LAYERS = 19
exports.MAP_WIDTH = 100
exports.MAP_HEIGHT = 50
exports.SeqIntType = SeqInt
exports.SeqFloatType = SeqFloat
exports.SeqBoolType = SeqBool
exports.checkError = checkError
exports.takeError = takeError
exports.TribalGameConfig = TribalGameConfig;
exports.tribalGameConfig = tribalGameConfig;
exports.TribalConfig = TribalConfig;
exports.tribalConfig = tribalConfig;
exports.TribalEnvType = TribalEnv
exports.TribalEnv = newTribalEnv
exports.defaultMaxSteps = defaultMaxSteps
exports.defaultTribalConfig = defaultTribalConfig
