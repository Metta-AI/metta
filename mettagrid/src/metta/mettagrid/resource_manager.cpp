#include "resource_manager.hpp"

#include <algorithm>
#include <cassert>
#include <stdexcept>

ResourceManager::ResourceManager(Grid* grid) : _grid(grid) {
  assert(_grid != nullptr);
  // Initialize team maps with existing objects
  _update_team_maps();
}

void ResourceManager::step() {
  // Main processing method called once per step
  // This is where you would implement any resource-related logic
  // that needs to run after event processing but before action processing

  // Example: You could implement resource decay, resource generation,
  // resource sharing between agents, etc. here

  // For now, this is a placeholder that can be extended
}

// Team-based object tracking methods
void ResourceManager::register_agent(Agent* agent) {
  if (!agent) return;

  unsigned int team_id = agent->group;
  _agents_by_team[team_id].push_back(agent);
}

void ResourceManager::register_converter(Converter* converter) {
  if (!converter) return;

  // Converters are neutral objects, assign them to team 0 (neutral team)
  unsigned int team_id = 0;
  _converters_by_team[team_id].push_back(converter);
}

void ResourceManager::unregister_agent(GridObjectId agent_id) {
  Agent* agent = _get_agent(agent_id);
  if (!agent) return;

  unsigned int team_id = agent->group;
  auto& team_agents = _agents_by_team[team_id];
  team_agents.erase(
    std::remove(team_agents.begin(), team_agents.end(), agent),
    team_agents.end()
  );

  // Remove empty team entry
  if (team_agents.empty()) {
    _agents_by_team.erase(team_id);
  }
}

void ResourceManager::unregister_converter(GridObjectId converter_id) {
  Converter* converter = _get_converter(converter_id);
  if (!converter) return;

  // Converters are neutral objects, assigned to team 0
  unsigned int team_id = 0;
  auto& team_converters = _converters_by_team[team_id];
  team_converters.erase(
    std::remove(team_converters.begin(), team_converters.end(), converter),
    team_converters.end()
  );

  // Remove empty team entry
  if (team_converters.empty()) {
    _converters_by_team.erase(team_id);
  }
}

// Team-based access methods
const std::vector<Agent*>& ResourceManager::get_agents_by_team(unsigned int team_id) const {
  static const std::vector<Agent*> empty_vector;
  auto it = _agents_by_team.find(team_id);
  return (it != _agents_by_team.end()) ? it->second : empty_vector;
}

const std::vector<Converter*>& ResourceManager::get_converters_by_team(unsigned int team_id) const {
  static const std::vector<Converter*> empty_vector;
  auto it = _converters_by_team.find(team_id);
  return (it != _converters_by_team.end()) ? it->second : empty_vector;
}

std::vector<unsigned int> ResourceManager::get_all_teams() const {
  std::vector<unsigned int> teams;
  for (const auto& [team_id, _] : _agents_by_team) {
    teams.push_back(team_id);
  }
  for (const auto& [team_id, _] : _converters_by_team) {
    if (std::find(teams.begin(), teams.end(), team_id) == teams.end()) {
      teams.push_back(team_id);
    }
  }
  return teams;
}

InventoryDelta ResourceManager::modify_agent_inventory(GridObjectId agent_id, InventoryItem item, InventoryDelta delta) {
  Agent* agent = _get_agent(agent_id);
  if (!agent) {
    throw std::runtime_error("Agent with id " + std::to_string(agent_id) + " not found");
  }

  return agent->update_inventory(item, delta);
}

InventoryQuantity ResourceManager::get_agent_inventory(GridObjectId agent_id, InventoryItem item) const {
  const Agent* agent = _get_agent(agent_id);
  if (!agent) {
    throw std::runtime_error("Agent with id " + std::to_string(agent_id) + " not found");
  }

  auto it = agent->inventory.find(item);
  return (it != agent->inventory.end()) ? it->second : 0;
}

const std::map<InventoryItem, InventoryQuantity>& ResourceManager::get_agent_inventory(GridObjectId agent_id) const {
  const Agent* agent = _get_agent(agent_id);
  if (!agent) {
    throw std::runtime_error("Agent with id " + std::to_string(agent_id) + " not found");
  }

  return agent->inventory;
}

InventoryDelta ResourceManager::modify_converter_inventory(GridObjectId converter_id, InventoryItem item, InventoryDelta delta) {
  Converter* converter = _get_converter(converter_id);
  if (!converter) {
    throw std::runtime_error("Converter with id " + std::to_string(converter_id) + " not found");
  }

  return converter->update_inventory(item, delta);
}

InventoryQuantity ResourceManager::get_converter_inventory(GridObjectId converter_id, InventoryItem item) const {
  const Converter* converter = _get_converter(converter_id);
  if (!converter) {
    throw std::runtime_error("Converter with id " + std::to_string(converter_id) + " not found");
  }

  auto it = converter->inventory.find(item);
  return (it != converter->inventory.end()) ? it->second : 0;
}

const std::map<InventoryItem, InventoryQuantity>& ResourceManager::get_converter_inventory(GridObjectId converter_id) const {
  const Converter* converter = _get_converter(converter_id);
  if (!converter) {
    throw std::runtime_error("Converter with id " + std::to_string(converter_id) + " not found");
  }

  return converter->inventory;
}

std::vector<GridObjectId> ResourceManager::get_all_agent_ids() const {
  std::vector<GridObjectId> agent_ids;

  for (unsigned int obj_id = 1; obj_id < _grid->objects.size(); obj_id++) {
    auto obj = _grid->object(obj_id);
    if (obj && dynamic_cast<Agent*>(obj)) {
      agent_ids.push_back(obj_id);
    }
  }

  return agent_ids;
}

std::vector<GridObjectId> ResourceManager::get_all_converter_ids() const {
  std::vector<GridObjectId> converter_ids;

  for (unsigned int obj_id = 1; obj_id < _grid->objects.size(); obj_id++) {
    auto obj = _grid->object(obj_id);
    if (obj && dynamic_cast<Converter*>(obj)) {
      converter_ids.push_back(obj_id);
    }
  }

  return converter_ids;
}

bool ResourceManager::is_agent(GridObjectId id) const {
  auto obj = _grid->object(id);
  return obj && dynamic_cast<Agent*>(obj) != nullptr;
}

bool ResourceManager::is_converter(GridObjectId id) const {
  auto obj = _grid->object(id);
  return obj && dynamic_cast<Converter*>(obj) != nullptr;
}

InventoryDelta ResourceManager::transfer_resource(GridObjectId from_id, GridObjectId to_id, InventoryItem item, InventoryQuantity amount) {
  // Get the source object
  auto from_obj = _grid->object(from_id);
  auto to_obj = _grid->object(to_id);

  if (!from_obj || !to_obj) {
    throw std::runtime_error("Invalid object IDs for resource transfer");
  }

  // Check if both objects have inventories
  Agent* from_agent = dynamic_cast<Agent*>(from_obj);
  Converter* from_converter = dynamic_cast<Converter*>(from_obj);
  Agent* to_agent = dynamic_cast<Agent*>(to_obj);
  Converter* to_converter = dynamic_cast<Converter*>(to_obj);

  if ((!from_agent && !from_converter) || (!to_agent && !to_converter)) {
    throw std::runtime_error("One or both objects do not have inventories");
  }

  // Get available amount from source
  InventoryQuantity available = 0;
  if (from_agent) {
    available = get_agent_inventory(from_id, item);
  } else if (from_converter) {
    available = get_converter_inventory(from_id, item);
  }

  // Calculate actual transfer amount
  InventoryQuantity transfer_amount = std::min(amount, available);
  if (transfer_amount == 0) {
    return 0;
  }

  // Remove from source
  InventoryDelta removed = 0;
  if (from_agent) {
    removed = modify_agent_inventory(from_id, item, -static_cast<InventoryDelta>(transfer_amount));
  } else if (from_converter) {
    removed = modify_converter_inventory(from_id, item, -static_cast<InventoryDelta>(transfer_amount));
  }

  // Add to destination
  if (to_agent) {
    modify_agent_inventory(to_id, item, static_cast<InventoryDelta>(transfer_amount));
  } else if (to_converter) {
    modify_converter_inventory(to_id, item, static_cast<InventoryDelta>(transfer_amount));
  }

  // Return the actual amount transferred (should be the same as removed)
  return removed;
}

Agent* ResourceManager::_get_agent(GridObjectId agent_id) const {
  auto obj = _grid->object(agent_id);
  return dynamic_cast<Agent*>(obj);
}

Converter* ResourceManager::_get_converter(GridObjectId converter_id) const {
  auto obj = _grid->object(converter_id);
  return dynamic_cast<Converter*>(obj);
}

void ResourceManager::_validate_object_id(GridObjectId id) const {
  if (id == 0 || id >= _grid->objects.size()) {
    throw std::runtime_error("Invalid object ID: " + std::to_string(id));
  }
}

// Team-based resource operations
InventoryQuantity ResourceManager::get_team_total_inventory(unsigned int team_id, InventoryItem item) const {
  InventoryQuantity total = 0;

  // Sum from agents
  const auto& agents = get_agents_by_team(team_id);
  for (const Agent* agent : agents) {
    auto it = agent->inventory.find(item);
    if (it != agent->inventory.end()) {
      total += it->second;
    }
  }

  // Sum from converters
  const auto& converters = get_converters_by_team(team_id);
  for (const Converter* converter : converters) {
    auto it = converter->inventory.find(item);
    if (it != converter->inventory.end()) {
      total += it->second;
    }
  }

  return total;
}

void ResourceManager::distribute_resources_to_team(unsigned int team_id, InventoryItem item, InventoryQuantity total_amount) {
  const auto& agents = get_agents_by_team(team_id);
  if (agents.empty()) return;

  // Distribute evenly among agents
  InventoryQuantity per_agent = total_amount / agents.size();
  InventoryQuantity remainder = total_amount % agents.size();

  for (size_t i = 0; i < agents.size(); ++i) {
    InventoryQuantity amount = per_agent + (i < remainder ? 1 : 0);
    if (amount > 0) {
      modify_agent_inventory(agents[i]->id, item, static_cast<InventoryDelta>(amount));
    }
  }
}

void ResourceManager::_update_team_maps() {
  // Clear existing maps
  _agents_by_team.clear();
  _converters_by_team.clear();

  // Rebuild maps from current grid state
  for (size_t i = 0; i < _grid->objects.size(); ++i) {
    GridObject* obj = _grid->objects[i].get();
    if (!obj) continue;

    if (Agent* agent = dynamic_cast<Agent*>(obj)) {
      _agents_by_team[agent->group].push_back(agent);
    } else if (Converter* converter = dynamic_cast<Converter*>(obj)) {
      // Converters are neutral objects, assign them to team 0
      _converters_by_team[0].push_back(converter);
    }
  }
}

void ResourceManager::on_inventory_changed(GridObjectId object_id, InventoryItem item, InventoryDelta delta) {
  // This method is called whenever an object's inventory changes
  // You can implement any logic here that needs to respond to inventory changes

  // Example: Log inventory changes, update team totals, trigger events, etc.
  // For now, this is a placeholder that can be extended

  // You could add logic like:
  // - Update team resource totals
  // - Trigger resource sharing between team members
  // - Log inventory changes for debugging
  // - Update statistics or metrics
}
