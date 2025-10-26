#include "supervisors/agent_supervisor.hpp"

#include <iostream>
#include <string>

#include "core/grid_object.hpp"  // for ObservationTokens
#include "objects/agent.hpp"
#include "systems/stats_tracker.hpp"

AgentSupervisor::AgentSupervisor(const AgentSupervisorConfig& config, Grid* grid, Agent* agent)
    : can_override_action_(config.can_override_action), name_(config.name), grid_(grid), agent_(agent) {}

void AgentSupervisor::supervise(ActionType* agent_action, ActionArg* agent_arg, const ObservationTokens& observation) {
  // Get the supervisor's recommended action
  auto [supervisor_action, supervisor_arg] = get_recommended_action(observation);

  // Record statistics about agreement/disagreement
  bool agrees = (supervisor_action == *agent_action && supervisor_arg == *agent_arg);

  if (agrees) {
    agent_->stats.incr(name_ + ".agrees");
  } else {
    agent_->stats.incr(name_ + ".disagrees");
  }

  if (can_override_action_ && !agrees) {
    agent_->stats.incr(name_ + ".override");
    *agent_action = supervisor_action;
    *agent_arg = supervisor_arg;
  }
}
