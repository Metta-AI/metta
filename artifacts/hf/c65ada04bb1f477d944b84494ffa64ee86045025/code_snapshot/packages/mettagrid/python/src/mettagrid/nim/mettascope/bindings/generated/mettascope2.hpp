#ifndef INCLUDE_METTASCOPE2_H
#define INCLUDE_METTASCOPE2_H

#include <stdint.h>

struct ActionRequest;

struct RenderResponse;

struct ActionRequest {
  int64_t agent_id;
  int64_t action_id;
  int64_t argument;
};

struct RenderResponse {

  private:

  uint64_t reference;

  public:

  bool getShouldClose();
  void setShouldClose(bool value);

  void free();

};

extern "C" {

ActionRequest mettascope2_action_request(int64_t agent_id, int64_t action_id, int64_t argument);

char mettascope2_action_request_eq(ActionRequest a, ActionRequest b);

void mettascope2_render_response_unref(RenderResponse render_response);

bool mettascope2_render_response_get_should_close(RenderResponse render_response);

void mettascope2_render_response_set_should_close(RenderResponse render_response, bool value);

int64_t mettascope2_render_response_actions_len(RenderResponse render_response);

ActionRequest mettascope2_render_response_actions_get(RenderResponse render_response, int64_t index);

void mettascope2_render_response_actions_set(RenderResponse render_response, int64_t index, ActionRequest value);

void mettascope2_render_response_actions_delete(RenderResponse render_response, int64_t index);

void mettascope2_render_response_actions_add(RenderResponse render_response, ActionRequest value);

void mettascope2_render_response_actions_clear(RenderResponse render_response);

RenderResponse mettascope2_init(const char* data_dir, const char* replay);

RenderResponse mettascope2_render(int64_t current_step, const char* replay_step);

}

ActionRequest actionRequest(int64_t agentId, int64_t actionId, int64_t argument) {
  return mettascope2_action_request(agentId, actionId, argument);
};

bool RenderResponse::getShouldClose(){
  return mettascope2_render_response_get_should_close(*this);
}

void RenderResponse::setShouldClose(bool value){
  mettascope2_render_response_set_should_close(*this, value);
}

void RenderResponse::free(){
  mettascope2_render_response_unref(*this);
}

RenderResponse init(const char* dataDir, const char* replay) {
  return mettascope2_init(dataDir, replay);
};

RenderResponse render(int64_t currentStep, const char* replayStep) {
  return mettascope2_render(currentStep, replayStep);
};

#endif
