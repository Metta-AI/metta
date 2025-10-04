#ifndef INCLUDE_METTASCOPE2_H
#define INCLUDE_METTASCOPE2_H

typedef struct ActionRequest {
  long long agent_id;
  long long action_id;
  long long argument;
} ActionRequest;

typedef long long RenderResponse;

ActionRequest mettascope2_action_request(long long agent_id, long long action_id, long long argument);

char mettascope2_action_request_eq(ActionRequest a, ActionRequest b);

void mettascope2_render_response_unref(RenderResponse render_response);

char mettascope2_render_response_get_should_close(RenderResponse render_response);

void mettascope2_render_response_set_should_close(RenderResponse render_response, char value);

long long mettascope2_render_response_actions_len(RenderResponse render_response);

ActionRequest mettascope2_render_response_actions_get(RenderResponse render_response, long long index);

void mettascope2_render_response_actions_set(RenderResponse render_response, long long index, ActionRequest value);

void mettascope2_render_response_actions_delete(RenderResponse render_response, long long index);

void mettascope2_render_response_actions_add(RenderResponse render_response, ActionRequest value);

void mettascope2_render_response_actions_clear(RenderResponse render_response);

RenderResponse mettascope2_init(char* data_dir, char* replay);

RenderResponse mettascope2_render(long long current_step, char* replay_step);

#endif
