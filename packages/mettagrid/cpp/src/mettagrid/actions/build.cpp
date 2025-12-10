#include "actions/build.hpp"

#include "config/mettagrid_config.hpp"
#include "objects/assembler.hpp"
#include "objects/chest.hpp"
#include "objects/wall.hpp"
#include "systems/observation_encoder.hpp"

// Implementation of Build::_create_object
// This is in a .cpp file to avoid circular header dependencies,
// as it needs to include all object types and configs.
GridObject* Build::_create_object(const std::string& object_key, const GridLocation& location) {
  // Check if game_config is valid
  if (_game_config == nullptr) {
    return nullptr;  // No game config
  }

  // Look up the object config from game config
  auto it = _game_config->objects.find(object_key);
  if (it == _game_config->objects.end()) {
    return nullptr;  // Object key not found
  }

  const GridObjectConfig* object_cfg = it->second.get();
  GridObject* new_object = nullptr;

  // Try to create based on config type
  // Wall
  if (const WallConfig* wall_config = dynamic_cast<const WallConfig*>(object_cfg)) {
    new_object = new Wall(location.r, location.c, *wall_config);
  }
  // Assembler (covers charger, converter, generator, etc.)
  else if (const AssemblerConfig* assembler_config = dynamic_cast<const AssemblerConfig*>(object_cfg)) {
    Assembler* assembler = new Assembler(location.r, location.c, *assembler_config, _stats_tracker);
    assembler->set_grid(_grid);
    // Full initialization like map-placed assemblers
    if (_current_timestep_ptr) {
      assembler->set_current_timestep_ptr(_current_timestep_ptr);
    }
    if (_obs_encoder) {
      assembler->set_obs_encoder(_obs_encoder);
    }
    if (_num_agents > 0) {
      assembler->init_agent_tracking(_num_agents);
    }
    new_object = assembler;
  }
  // Chest
  else if (const ChestConfig* chest_config = dynamic_cast<const ChestConfig*>(object_cfg)) {
    Chest* chest = new Chest(location.r, location.c, *chest_config, _stats_tracker);
    chest->set_grid(_grid);
    if (_obs_encoder) {
      chest->set_obs_encoder(_obs_encoder);
    }
    new_object = chest;
  }

  if (new_object) {
    if (_grid->add_object(new_object)) {
      return new_object;
    } else {
      delete new_object;
      return nullptr;
    }
  }

  return nullptr;
}
