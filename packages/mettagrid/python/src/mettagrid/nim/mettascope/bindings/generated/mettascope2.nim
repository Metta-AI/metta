import bumpy, chroma, unicode, vmath

export bumpy, chroma, unicode, vmath

when defined(windows):
  const libName = "mettascope2.dll"
elif defined(macosx):
  const libName = "libmettascope2.dylib"
else:
  const libName = "libmettascope2.so"

{.push dynlib: libName.}

type Mettascope2Error = object of ValueError

type ActionRequest* = object
  agentId*: int
  actionId*: int
  argument*: int

proc actionRequest*(agent_id: int, action_id: int, argument: int): ActionRequest =
  result.agent_id = agent_id
  result.action_id = action_id
  result.argument = argument

type RenderResponseObj = object
  reference: pointer

type RenderResponse* = ref RenderResponseObj

proc mettascope2_render_response_unref(x: RenderResponseObj) {.importc: "mettascope2_render_response_unref", cdecl.}

proc `=destroy`(x: var RenderResponseObj) =
  mettascope2_render_response_unref(x)

proc mettascope2_render_response_get_should_close(renderResponse: RenderResponse): bool {.importc: "mettascope2_render_response_get_should_close", cdecl.}

proc shouldClose*(renderResponse: RenderResponse): bool {.inline.} =
  mettascope2_render_response_get_should_close(renderResponse)

proc mettascope2_render_response_set_should_close(renderResponse: RenderResponse, shouldClose: bool) {.importc: "mettascope2_render_response_set_should_close", cdecl.}

proc `shouldClose=`*(renderResponse: RenderResponse, shouldClose: bool) =
  mettascope2_render_response_set_should_close(renderResponse, shouldClose)

type RenderResponseActions = object
    renderResponse: RenderResponse

proc actions*(renderResponse: RenderResponse): RenderResponseActions =
  RenderResponseActions(renderResponse: renderResponse)

proc mettascope2_render_response_actions_len(s: RenderResponse): int {.importc: "mettascope2_render_response_actions_len", cdecl.}

proc len*(s: RenderResponseActions): int =
  mettascope2_render_response_actions_len(s.renderResponse)

proc mettascope2_render_response_actions_add(s: RenderResponse, v: ActionRequest) {.importc: "mettascope2_render_response_actions_add", cdecl.}

proc add*(s: RenderResponseActions, v: ActionRequest) =
  mettascope2_render_response_actions_add(s.renderResponse, v)

proc mettascope2_render_response_actions_get(s: RenderResponse, i: int): ActionRequest {.importc: "mettascope2_render_response_actions_get", cdecl.}

proc `[]`*(s: RenderResponseActions, i: int): ActionRequest =
  mettascope2_render_response_actions_get(s.renderResponse, i)

proc mettascope2_render_response_actions_set(s: RenderResponse, i: int, v: ActionRequest) {.importc: "mettascope2_render_response_actions_set", cdecl.}

proc `[]=`*(s: RenderResponseActions, i: int, v: ActionRequest) =
  mettascope2_render_response_actions_set(s.renderResponse, i, v)

proc mettascope2_render_response_actions_delete(s: RenderResponse, i: int) {.importc: "mettascope2_render_response_actions_delete", cdecl.}

proc delete*(s: RenderResponseActions, i: int) =
  mettascope2_render_response_actions_delete(s.renderResponse, i)

proc mettascope2_render_response_actions_clear(s: RenderResponse) {.importc: "mettascope2_render_response_actions_clear", cdecl.}

proc clear*(s: RenderResponseActions) =
  mettascope2_render_response_actions_clear(s.renderResponse)

proc mettascope2_init(data_dir: cstring, replay: cstring): RenderResponse {.importc: "mettascope2_init", cdecl.}

proc init*(dataDir: string, replay: string): RenderResponse {.inline.} =
  result = mettascope2_init(dataDir.cstring, replay.cstring)

proc mettascope2_render(current_step: int, replay_step: cstring): RenderResponse {.importc: "mettascope2_render", cdecl.}

proc render*(currentStep: int, replayStep: string): RenderResponse {.inline.} =
  result = mettascope2_render(currentStep, replayStep.cstring)

