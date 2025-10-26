#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_SUPERVISORS_SUPERVISOR_FACTORY_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_SUPERVISORS_SUPERVISOR_FACTORY_HPP_

#include <memory>
#include <string>

#include "actions/action_handler.hpp"
#include "supervisors/agent_supervisor.hpp"
#include "supervisors/patrol_supervisor.hpp"

// Factory class to create supervisors from configurations
using ActionHandlers = std::vector<std::unique_ptr<ActionHandler>>;

class SupervisorFactory {
public:
  static std::unique_ptr<AgentSupervisor> create(const AgentSupervisorConfig* config, Grid* grid, Agent* agent) {
    if (!config) {
      return nullptr;
    }

    // Check if it's a PatrolSupervisorConfig
    if (auto* patrol_config = dynamic_cast<const PatrolSupervisorConfig*>(config)) {
      return std::make_unique<PatrolSupervisor>(*patrol_config, grid, agent);
    }

    // Default: create base supervisor (which would be abstract, so this shouldn't happen)
    // In practice, you'd have a concrete default supervisor or throw an error
    return nullptr;
  }
};

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_SUPERVISORS_SUPERVISOR_FACTORY_HPP_
