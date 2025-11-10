#ifndef INCLUDE_HEURISTICAGENTS_H
#define INCLUDE_HEURISTICAGENTS_H

typedef long long HeuristicAgent;

void heuristic_agents_heuristic_agent_unref(HeuristicAgent heuristic_agent);

HeuristicAgent heuristic_agents_new_heuristic_agent(long long agent_id, char* environment_config);

long long heuristic_agents_heuristic_agent_get_agent_id(HeuristicAgent heuristic_agent);

void heuristic_agents_heuristic_agent_set_agent_id(HeuristicAgent heuristic_agent, long long value);

void heuristic_agents_heuristic_agent_reset(HeuristicAgent agent);

void heuristic_agents_heuristic_agent_step(HeuristicAgent agent, long long num_agents, long long num_tokens, long long size_token, void* row_observations, long long num_actions, void* raw_actions);

void heuristic_agents_init_chook();

#endif
