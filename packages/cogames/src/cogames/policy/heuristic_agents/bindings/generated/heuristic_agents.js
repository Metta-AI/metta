var ffi = require('ffi-napi');
var Struct = require("ref-struct-napi");
var ArrayType = require('ref-array-napi');

var dll = {};

function HeuristicAgentsException(message) {
  this.message = message;
  this.name = 'HeuristicAgentsException';
}

HeuristicAgent = Struct({'nimRef': 'uint64'});
HeuristicAgent.prototype.isNull = function(){
  return this.nimRef == 0;
};
HeuristicAgent.prototype.isEqual = function(other){
  return this.nimRef == other.nimRef;
};
HeuristicAgent.prototype.unref = function(){
  return dll.heuristic_agents_heuristic_agent_unref(this)
};
function newHeuristicAgent(agent_id, environment_config){
  var result = dll.heuristic_agents_new_heuristic_agent(agent_id, environment_config)
  const registry = new FinalizationRegistry(function(obj) {
    console.log("js unref")
    obj.unref()
  });
  registry.register(result, null);
  return result
}
Object.defineProperty(HeuristicAgent.prototype, 'agentId', {
  get: function() {return dll.heuristic_agents_heuristic_agent_get_agent_id(this)},
  set: function(v) {dll.heuristic_agents_heuristic_agent_set_agent_id(this, v)}
});

HeuristicAgent.prototype.reset = function(){
  dll.heuristic_agents_heuristic_agent_reset(this)
}

HeuristicAgent.prototype.step = function(num_agents, num_tokens, size_token, row_observations, num_actions, raw_actions){
  dll.heuristic_agents_heuristic_agent_step(this, num_agents, num_tokens, size_token, row_observations, num_actions, raw_actions)
}

function initCHook(){
  dll.heuristic_agents_init_chook()
}


var dllPath = ""
if(process.platform == "win32") {
  dllPath = __dirname + '/heuristic_agents.dll'
} else if (process.platform == "darwin") {
  dllPath = __dirname + '/libheuristic_agents.dylib'
} else {
  dllPath = __dirname + '/libheuristic_agents.so'
}

dll = ffi.Library(dllPath, {
  'heuristic_agents_heuristic_agent_unref': ['void', [HeuristicAgent]],
  'heuristic_agents_new_heuristic_agent': [HeuristicAgent, ['int64', 'string']],
  'heuristic_agents_heuristic_agent_get_agent_id': ['int64', [HeuristicAgent]],
  'heuristic_agents_heuristic_agent_set_agent_id': ['void', [HeuristicAgent, 'int64']],
  'heuristic_agents_heuristic_agent_reset': ['void', [HeuristicAgent]],
  'heuristic_agents_heuristic_agent_step': ['void', [HeuristicAgent, 'int64', 'int64', 'int64', pointer, 'int64', pointer]],
  'heuristic_agents_init_chook': ['void', []],
});

exports.HeuristicAgentType = HeuristicAgent
exports.HeuristicAgent = newHeuristicAgent
exports.initCHook = initCHook
