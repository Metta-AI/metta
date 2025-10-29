#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_SUPERVISORS_AGENT_SUPERVISOR_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_SUPERVISORS_AGENT_SUPERVISOR_HPP_

#include <memory>
#include <string>
#include <utility>  // for std::pair
#include <vector>

#include "core/grid_object.hpp"  // for ObservationTokens
#include "core/types.hpp"

// Forward declarations
class Grid;
class Agent;
class ActionHandler;

// Configuration for AgentSupervisor
struct AgentSupervisorConfig {
  // Whether the supervisor can override the agent's action
  bool can_override_action;
  // Name of the supervisor for logging
  std::string name;

  explicit AgentSupervisorConfig(bool can_override_action = false, const std::string& name = "supervisor")
      : can_override_action(can_override_action), name(name) {}

  virtual ~AgentSupervisorConfig() = default;
};

// Base class for agent supervisors
class AgentSupervisor {
public:
  explicit AgentSupervisor(const AgentSupervisorConfig& config, Grid* grid, Agent* agent);

  virtual ~AgentSupervisor() = default;

  // Called before the agent's action is executed
  // The supervisor receives the same observation data as the agent
  void supervise(ActionType* agent_action, ActionArg* agent_arg, const ObservationTokens& observation);

  // Called after the action has been executed
  virtual void post_action(bool action_success) {}

protected:
  // Subclasses must implement this to provide their recommended action
  virtual std::pair<ActionType, ActionArg> get_recommended_action(const ObservationTokens& observation) = 0;

  bool can_override_action_;
  std::string name_;
  Grid* grid_;
  Agent* agent_;
};

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_SUPERVISORS_AGENT_SUPERVISOR_HPP_
