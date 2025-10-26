#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_SUPERVISORS_PATROL_SUPERVISOR_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_SUPERVISORS_PATROL_SUPERVISOR_HPP_

#include "actions/orientation.hpp"
#include "objects/agent.hpp"
#include "supervisors/agent_supervisor.hpp"
struct PatrolSupervisorConfig : public AgentSupervisorConfig {
  // Number of steps to move in each direction before turning
  int steps_per_direction;

  PatrolSupervisorConfig(int steps_per_direction = 5,
                         bool can_override_action = false,
                         const std::string& name = "patrol_supervisor")
      : AgentSupervisorConfig(can_override_action, name), steps_per_direction(steps_per_direction) {}
};

// Supervisor that makes the agent patrol left and right
class PatrolSupervisor : public AgentSupervisor {
public:
  explicit PatrolSupervisor(const PatrolSupervisorConfig& config, Grid* grid, Agent* agent)
      : AgentSupervisor(config, grid, agent), config_(config), steps_in_current_direction_(0) {}

protected:
  std::pair<ActionType, ActionArg> get_recommended_action(const ObservationTokens& observation) override {
    // Unused parameter
    (void)observation;

    // Check if we need to change direction
    if (steps_in_current_direction_ >= config_.steps_per_direction) {
      moving_right_ = !moving_right_;
      steps_in_current_direction_ = 0;
    }

    // Increment step counter
    steps_in_current_direction_++;

    // Return move action with appropriate direction
    ActionType move_action = 1;
    ActionArg direction = moving_right_ ? Orientation::East : Orientation::West;

    return {move_action, direction};
  }

private:
  PatrolSupervisorConfig config_;
  bool moving_right_;
  int steps_in_current_direction_;
};

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_SUPERVISORS_PATROL_SUPERVISOR_HPP_
