var ffi = require('ffi-napi');
var Struct = require("ref-struct-napi");
var ArrayType = require('ref-array-napi');

var dll = {};

function Mettascope2Exception(message) {
  this.message = message;
  this.name = 'Mettascope2Exception';
}

const ActionRequest = Struct({
  'agentId':'int64',
  'actionId':'int64',
  'argument':'int64'
})
actionRequest = function(agent_id, action_id, argument){
  var v = new ActionRequest();
  v.agent_id = agent_id
  v.action_id = action_id
  v.argument = argument
  return v;
}
ActionRequest.prototype.isEqual = function(other){
  return self.agentId == other.agentId && self.actionId == other.actionId && self.argument == other.argument;
};

RenderResponse = Struct({'nimRef': 'uint64'});
RenderResponse.prototype.isNull = function(){
  return this.nimRef == 0;
};
RenderResponse.prototype.isEqual = function(other){
  return this.nimRef == other.nimRef;
};
RenderResponse.prototype.unref = function(){
  return dll.mettascope2_render_response_unref(this)
};
Object.defineProperty(RenderResponse.prototype, 'shouldClose', {
  get: function() {return dll.mettascope2_render_response_get_should_close(this)},
  set: function(v) {dll.mettascope2_render_response_set_should_close(this, v)}
});
function RenderResponseActions(renderResponse){
  this.renderResponse = renderResponse;
}
RenderResponseActions.prototype.length = function(){
  return dll.mettascope2_render_response_actions_len(this.render_response)
};
RenderResponseActions.prototype.get = function(index){
  return dll.mettascope2_render_response_actions_get(this.render_response, index)
};
RenderResponseActions.prototype.set = function(index, value){
  dll.mettascope2_render_response_actions_set(this.render_response, index, value)
};
RenderResponseActions.prototype.delete = function(index){
  dll.mettascope2_render_response_actions_delete(this.render_response, index)
};
RenderResponseActions.prototype.add = function(value){
  dll.mettascope2_render_response_actions_add(this.render_response, value)
};
RenderResponseActions.prototype.clear = function(){
  dll.mettascope2_render_response_actions_clear(this.render_response)
};
Object.defineProperty(RenderResponse.prototype, 'actions', {
  get: function() {return new RenderResponseActions(this)},
});

function init(data_dir, replay){
  result = dll.mettascope2_init(data_dir, replay)
  return result
}

function render(current_step, replay_step){
  result = dll.mettascope2_render(current_step, replay_step)
  return result
}


var dllPath = ""
if(process.platform == "win32") {
  dllPath = __dirname + '/mettascope2.dll'
} else if (process.platform == "darwin") {
  dllPath = __dirname + '/libmettascope2.dylib'
} else {
  dllPath = __dirname + '/libmettascope2.so'
}

dll = ffi.Library(dllPath, {
  'mettascope2_render_response_unref': ['void', [RenderResponse]],
  'mettascope2_render_response_get_should_close': ['bool', [RenderResponse]],
  'mettascope2_render_response_set_should_close': ['void', [RenderResponse, 'bool']],
  'mettascope2_render_response_actions_len': ['uint64', [RenderResponse]],
  'mettascope2_render_response_actions_get': [ActionRequest, [RenderResponse, 'uint64']],
  'mettascope2_render_response_actions_set': ['void', [RenderResponse, 'uint64', ActionRequest]],
  'mettascope2_render_response_actions_delete': ['void', [RenderResponse, 'uint64']],
  'mettascope2_render_response_actions_add': ['void', [RenderResponse, ActionRequest]],
  'mettascope2_render_response_actions_clear': ['void', [RenderResponse]],
  'mettascope2_init': [RenderResponse, ['string', 'string']],
  'mettascope2_render': [RenderResponse, ['int64', 'string']],
});

exports.ActionRequest = ActionRequest;
exports.actionRequest = actionRequest;
exports.RenderResponseType = RenderResponse
exports.init = init
exports.render = render
