#include "actions/build.hpp"

#include "config/mettagrid_config.hpp"
#include "core/grid.hpp"
#include "objects/agent.hpp"
#include "objects/assembler.hpp"
#include "objects/chest.hpp"
#include "objects/wall.hpp"
#include "systems/observation_encoder.hpp"
#include "systems/stats_tracker.hpp"

Build::Build(const BuildActionConfig& cfg,
             const GameConfig* game_config,
             StatsTracker* stats_tracker,
             const std::string& action_name)
    : ActionHandler(cfg, action_name),
      _vibe_builds(cfg.vibe_builds),
      _enabled(cfg.enabled),
      _vibes(cfg.vibes),
      _game_config(game_config),
      _stats_tracker(stats_tracker),
      _current_timestep_ptr(nullptr),
      _obs_encoder(nullptr),
      _num_agents(0) {
  priority = 0;  // Lower priority (same as transfer)
}

void Build::set_runtime_context(unsigned int* current_timestep_ptr,
                                ObservationEncoder* obs_encoder,
                                unsigned int num_agents) {
  _current_timestep_ptr = current_timestep_ptr;
  _obs_encoder = obs_encoder;
  _num_agents = num_agents;
}

std::vector<Action> Build::create_actions() {
  // Build doesn't create standalone actions - it's triggered by move
  return {};
}

const std::vector<ObservationType>& Build::get_vibes() const {
  return _vibes;
}

bool Build::has_build_for_vibe(ObservationType vibe) const {
  return _vibe_builds.find(vibe) != _vibe_builds.end();
}

bool Build::try_build(Agent& actor, const GridLocation& previous_location) {
  const std::string& actor_group = actor.group_name;

  // Debug: track that build was attempted
  actor.stats.incr(_action_prefix(actor_group) + "attempted");
  actor.stats.incr(_action_prefix(actor_group) + "vibe." + std::to_string(actor.vibe));

  auto vibe_it = _vibe_builds.find(actor.vibe);
  if (vibe_it == _vibe_builds.end()) {
    actor.stats.incr(_action_prefix(actor_group) + "failed.no_build_for_vibe");
    return false;  // No build configured for this vibe
  }

  const VibeBuildEffect& effect = vibe_it->second;

  // 1. Check if the previous location is empty (it should be since we just moved)
  if (!_grid->is_empty(previous_location.r, previous_location.c)) {
    return false;  // Location is not empty (shouldn't happen normally)
  }

  // 2. Check if actor has resources to pay the cost
  for (const auto& [resource, amount] : effect.cost) {
    if (actor.inventory.amount(resource) < amount) {
      actor.stats.incr(_action_prefix(actor_group) + "failed.insufficient_resources");
      return false;  // Actor doesn't have enough resources
    }
  }

  // 3. Try to create the object at the previous location
  GridObject* new_object = _create_object(effect.object_key, previous_location);
  if (!new_object) {
    actor.stats.incr(_action_prefix(actor_group) + "failed.object_creation");
    return false;  // Failed to create object
  }

  // 4. Deduct resources from actor
  for (const auto& [resource, amount] : effect.cost) {
    if (amount > 0) {
      InventoryDelta delta = static_cast<InventoryDelta>(-static_cast<int>(amount));
      InventoryDelta actual = actor.inventory.update(resource, delta);
      if (actual != 0) {
        _log_build_cost(actor, resource, -actual);
      }
    }
  }

  // 5. Log success
  actor.stats.incr(_action_prefix(actor_group) + "count");
  actor.stats.incr(_action_prefix(actor_group) + "built." + effect.object_key);

  // Track build stats by object name (agent and game level)
  actor.stats.incr("build." + effect.object_key);
  if (_stats_tracker) {
    _stats_tracker->incr("build." + effect.object_key);
  }

  return true;
}

bool Build::_handle_action(Agent& actor, ActionArg arg) {
  // Build is not called directly as an action
  (void)actor;
  (void)arg;
  return false;
}

std::string Build::_action_prefix(const std::string& group) const {
  return "action." + _action_name + "." + group + ".";
}

void Build::_log_build_cost(Agent& actor, InventoryItem item, InventoryDelta amount) const {
  const std::string& actor_group = actor.group_name;
  const std::string item_name = actor.stats.resource_name(item);

  actor.stats.add(_action_prefix(actor_group) + "cost." + item_name, amount);
}

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
