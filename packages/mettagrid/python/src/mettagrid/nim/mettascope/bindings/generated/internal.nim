when not defined(gcArc) and not defined(gcOrc):
  {.error: "Please use --gc:arc or --gc:orc when using Genny.".}

when (NimMajor, NimMinor, NimPatch) == (1, 6, 2):
  {.error: "Nim 1.6.2 not supported with Genny due to FFI issues.".}
proc mettascope2_action_request*(agent_id: int, action_id: int, argument: int): ActionRequest {.raises: [], cdecl, exportc, dynlib.} =
  result.agent_id = agent_id
  result.action_id = action_id
  result.argument = argument

proc mettascope2_action_request_eq*(a, b: ActionRequest): bool {.raises: [], cdecl, exportc, dynlib.}=
  a.agent_id == b.agent_id and a.action_id == b.action_id and a.argument == b.argument

proc mettascope2_render_response_unref*(x: RenderResponse) {.raises: [], cdecl, exportc, dynlib.} =
  GC_unref(x)

proc mettascope2_render_response_get_should_close*(render_response: RenderResponse): bool {.raises: [], cdecl, exportc, dynlib.} =
  render_response.shouldClose

proc mettascope2_render_response_set_should_close*(render_response: RenderResponse, shouldClose: bool) {.raises: [], cdecl, exportc, dynlib.} =
  render_response.shouldClose = shouldClose

proc mettascope2_render_response_actions_len*(render_response: RenderResponse): int {.raises: [], cdecl, exportc, dynlib.} =
  render_response.actions.len

proc mettascope2_render_response_actions_add*(render_response: RenderResponse, v: ActionRequest) {.raises: [], cdecl, exportc, dynlib.} =
  render_response.actions.add(v)

proc mettascope2_render_response_actions_get*(render_response: RenderResponse, i: int): ActionRequest {.raises: [], cdecl, exportc, dynlib.} =
  render_response.actions[i]

proc mettascope2_render_response_actions_set*(render_response: RenderResponse, i: int, v: ActionRequest) {.raises: [], cdecl, exportc, dynlib.} =
  render_response.actions[i] = v

proc mettascope2_render_response_actions_delete*(render_response: RenderResponse, i: int) {.raises: [], cdecl, exportc, dynlib.} =
  render_response.actions.delete(i)

proc mettascope2_render_response_actions_clear*(render_response: RenderResponse) {.raises: [], cdecl, exportc, dynlib.} =
  render_response.actions.setLen(0)

proc mettascope2_init*(data_dir: cstring, replay: cstring): RenderResponse {.raises: [], cdecl, exportc, dynlib.} =
  init(data_dir.`$`, replay.`$`)

proc mettascope2_render*(current_step: int, replay_step: cstring): RenderResponse {.raises: [], cdecl, exportc, dynlib.} =
  render(current_step, replay_step.`$`)

