#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_SUPERVISORS_PATROL_SUPERVISOR_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_SUPERVISORS_PATROL_SUPERVISOR_HPP_

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
  explicit PatrolSupervisor(const PatrolSupervisorConfig& config)
      : AgentSupervisor(config), config_(config), steps_in_current_direction_(0) {}

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
    // Assuming action 1 is move, and directions are:
    // 0 = North, 1 = East (right), 2 = South, 3 = West (left)
    ActionType move_action = 1;
    ActionArg direction = moving_right_ ? 1 : 3;  // East : West

    return {move_action, direction};
  }

  void on_supervise(ActionType agent_action,
                    ActionArg agent_arg,
                    ActionType supervisor_action,
                    ActionArg supervisor_arg,
                    bool agrees) override {
    if (!agrees) {
      agent_->stats.incr(name_ + ".disagree");
      if (moving_right_) {
        agent_->stats.incr(name_ + ".moving_right.disagree");
      } else {
        agent_->stats.incr(name_ + ".moving_left.disagree");
      }
    }
  }

private:
  PatrolSupervisorConfig config_;
  bool moving_right_;
  int steps_in_current_direction_;
};

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_SUPERVISORS_PATROL_SUPERVISOR_HPP_
