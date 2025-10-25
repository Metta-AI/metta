#include "supervisors/agent_supervisor.hpp"

#include <string>

#include "core/grid_object.hpp"  // for ObservationTokens
#include "objects/agent.hpp"
#include "systems/stats_tracker.hpp"

AgentSupervisor::AgentSupervisor(const AgentSupervisorConfig& config)
    : can_override_action_(config.can_override_action), name_(config.name), grid_(nullptr), agent_(nullptr) {}

void AgentSupervisor::init(Grid* grid, Agent* agent) {
  grid_ = grid;
  agent_ = agent;
}

std::pair<ActionType, ActionArg> AgentSupervisor::supervise(ActionType agent_action,
                                                            ActionArg agent_arg,
                                                            const ObservationTokens& observation) {
  // Get the supervisor's recommended action
  auto [supervisor_action, supervisor_arg] = get_recommended_action(observation);

  // Record statistics about agreement/disagreement
  bool agrees = (supervisor_action == agent_action && supervisor_arg == agent_arg);

  if (agrees) {
    agent_->stats.incr(name_ + ".right");
  } else {
    agent_->stats.incr(name_ + ".wrong");

    // Record specific disagreement types
    if (supervisor_action != agent_action) {
      agent_->stats.incr(name_ + ".wrong_action");
      agent_->stats.incr(name_ + ".action." + std::to_string(agent_action) + ".wrong");
    }
    if (supervisor_arg != agent_arg) {
      agent_->stats.incr(name_ + ".wrong_arg");
    }
  }

  // Additional statistics
  on_supervise(agent_action, agent_arg, supervisor_action, supervisor_arg, agrees);

  // Return either the agent's action or the supervisor's override
  if (can_override_action_ && !agrees) {
    agent_->stats.incr(name_ + ".override");
    return {supervisor_action, supervisor_arg};
  }

  return {agent_action, agent_arg};
}
