#include "core/grid_object_factory.hpp"

#include <memory>
#include <vector>

#include "core/grid.hpp"
#include "handler/handler.hpp"
#include "objects/agent.hpp"
#include "objects/agent_config.hpp"
#include "objects/assembler.hpp"
#include "objects/assembler_config.hpp"
#include "objects/chest.hpp"
#include "objects/chest_config.hpp"
#include "objects/wall.hpp"
#include "systems/observation_encoder.hpp"
#include "systems/stats_tracker.hpp"

namespace mettagrid {

GridObject* create_object_from_config(GridCoord r,
                                      GridCoord c,
                                      const GridObjectConfig* config,
                                      StatsTracker* stats,
                                      const std::vector<std::string>* resource_names,
                                      Grid* grid,
                                      const ObservationEncoder* obs_encoder,
                                      unsigned int* current_timestep_ptr) {
  GridObject* created_object = nullptr;

  // Try each config type in order
  // TODO: replace the dynamic casts with virtual dispatch

  const WallConfig* wall_config = dynamic_cast<const WallConfig*>(config);
  if (wall_config) {
    created_object = new Wall(r, c, *wall_config);
  }

  const AgentConfig* agent_config = dynamic_cast<const AgentConfig*>(config);
  if (!created_object && agent_config) {
    Agent* agent = new Agent(r, c, *agent_config, resource_names);
    agent->set_obs_encoder(obs_encoder);
    created_object = agent;
  }

  const AssemblerConfig* assembler_config = dynamic_cast<const AssemblerConfig*>(config);
  if (!created_object && assembler_config) {
    Assembler* assembler = new Assembler(r, c, *assembler_config, stats);
    assembler->set_grid(grid);
    assembler->set_current_timestep_ptr(current_timestep_ptr);
    assembler->set_obs_encoder(obs_encoder);
    created_object = assembler;
  }

  const ChestConfig* chest_config = dynamic_cast<const ChestConfig*>(config);
  if (!created_object && chest_config) {
    Chest* chest = new Chest(r, c, *chest_config, stats);
    chest->set_grid(grid);
    chest->set_obs_encoder(obs_encoder);
    created_object = chest;
  }

  // Handle base GridObjectConfig as a static object (similar to wall but can have AOEs)
  if (!created_object) {
    Wall* static_obj = new Wall(r, c, WallConfig(config->type_id, config->type_name, config->initial_vibe));
    static_obj->tag_ids = config->tag_ids;
    created_object = static_obj;
  }

  // Set up handlers for this object
  if (created_object) {
    // on_use handlers
    if (!config->on_use_handlers.empty()) {
      std::vector<std::shared_ptr<mettagrid::Handler>> handlers;
      handlers.reserve(config->on_use_handlers.size());
      for (const auto& handler_config : config->on_use_handlers) {
        handlers.push_back(std::make_shared<mettagrid::Handler>(handler_config));
      }
      created_object->set_on_use_handlers(std::move(handlers));
    }

    // on_update handlers
    if (!config->on_update_handlers.empty()) {
      std::vector<std::shared_ptr<mettagrid::Handler>> handlers;
      handlers.reserve(config->on_update_handlers.size());
      for (const auto& handler_config : config->on_update_handlers) {
        handlers.push_back(std::make_shared<mettagrid::Handler>(handler_config));
      }
      created_object->set_on_update_handlers(std::move(handlers));
    }

    // AOE handlers
    if (!config->aoe_handlers.empty()) {
      std::vector<std::shared_ptr<mettagrid::Handler>> handlers;
      handlers.reserve(config->aoe_handlers.size());
      for (const auto& handler_config : config->aoe_handlers) {
        handlers.push_back(std::make_shared<mettagrid::Handler>(handler_config));
      }
      created_object->set_aoe_handlers(std::move(handlers));
    }
  }

  return created_object;
}

}  // namespace mettagrid
