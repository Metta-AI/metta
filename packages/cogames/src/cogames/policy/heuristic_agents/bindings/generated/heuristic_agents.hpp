#ifndef INCLUDE_HEURISTICAGENTS_H
#define INCLUDE_HEURISTICAGENTS_H

#include <stdint.h>

struct HeuristicAgent;

struct HeuristicAgent {

  private:

  uint64_t reference;

  public:

  HeuristicAgent(int64_t agentId, const char* environmentConfig);

  int64_t getAgentId();
  void setAgentId(int64_t value);

  void free();

  void reset();

  void step(int64_t numAgents, int64_t numTokens, int64_t sizeToken, pointer rowObservations, int64_t numActions, pointer rawActions);

};

extern "C" {

void heuristic_agents_heuristic_agent_unref(HeuristicAgent heuristic_agent);

HeuristicAgent heuristic_agents_new_heuristic_agent(int64_t agent_id, const char* environment_config);

int64_t heuristic_agents_heuristic_agent_get_agent_id(HeuristicAgent heuristic_agent);

void heuristic_agents_heuristic_agent_set_agent_id(HeuristicAgent heuristic_agent, int64_t value);

void heuristic_agents_heuristic_agent_reset(HeuristicAgent agent);

void heuristic_agents_heuristic_agent_step(HeuristicAgent agent, int64_t num_agents, int64_t num_tokens, int64_t size_token, pointer row_observations, int64_t num_actions, pointer raw_actions);

void heuristic_agents_init_chook();

}

HeuristicAgent::HeuristicAgent(int64_t agentId, const char* environmentConfig) {
  this->reference = heuristic_agents_new_heuristic_agent(agentId, environmentConfig).reference;
}

int64_t HeuristicAgent::getAgentId(){
  return heuristic_agents_heuristic_agent_get_agent_id(*this);
}

void HeuristicAgent::setAgentId(int64_t value){
  heuristic_agents_heuristic_agent_set_agent_id(*this, value);
}

void HeuristicAgent::free(){
  heuristic_agents_heuristic_agent_unref(*this);
}

void HeuristicAgent::reset() {
  heuristic_agents_heuristic_agent_reset(*this);
};

void HeuristicAgent::step(int64_t numAgents, int64_t numTokens, int64_t sizeToken, pointer rowObservations, int64_t numActions, pointer rawActions) {
  heuristic_agents_heuristic_agent_step(*this, numAgents, numTokens, sizeToken, rowObservations, numActions, rawActions);
};

initCHook() {
  heuristic_agents_init_chook();
};

#endif
