#include "resource_manager.hpp"

#include <algorithm>
#include <cassert>
#include <random>
#include <stdexcept>

#include "objects/agent.hpp"
#include "objects/converter.hpp"

ResourceManager::ResourceManager(Grid* grid) : _grid(grid) {
  assert(_grid != nullptr);
  // Initialize group maps with existing objects
  _update_group_maps();
  // Initialize inventory totals
  _update_inventory_totals();
}

void ResourceManager::step() {
  // Main processing method called once per step
  // This is where you would implement any resource-related logic
  // that needs to run after event processing but before action processing

  // Example: You could implement resource decay, resource generation,
  // resource sharing between agents, etc. here

  // For now, this is a placeholder that can be extended
}

// Object registration and tracking methods
void ResourceManager::register_inventory_object(HasInventory* obj) {
  if (!obj) return;

  std::string group_name = _get_group_name(obj);
  _objects_by_group[group_name].push_back(obj);
}

void ResourceManager::unregister_inventory_object(GridObjectId object_id) {
  HasInventory* obj = _get_inventory_object(object_id);
  if (!obj) return;

  std::string group_name = _get_group_name(obj);
  auto& group_objects = _objects_by_group[group_name];
  group_objects.erase(
    std::remove(group_objects.begin(), group_objects.end(), obj),
    group_objects.end());

  // Remove empty group entry
  if (group_objects.empty()) {
    _objects_by_group.erase(group_name);
  }
}

// Group-based access methods
const std::vector<HasInventory*>& ResourceManager::get_objects_by_group(const std::string& group_name) const {
  static const std::vector<HasInventory*> empty_vector;
  auto it = _objects_by_group.find(group_name);
  return (it != _objects_by_group.end()) ? it->second : empty_vector;
}

std::vector<std::string> ResourceManager::get_all_groups() const {
  std::vector<std::string> groups;
  for (const auto& [group_name, _] : _objects_by_group) {
    groups.push_back(group_name);
  }
  return groups;
}

// Inventory management methods
InventoryDelta ResourceManager::modify_inventory(GridObjectId object_id, InventoryItem item, InventoryDelta delta) {
  HasInventory* obj = _get_inventory_object(object_id);
  if (!obj) {
    throw std::runtime_error("Object not found: " + std::to_string(object_id));
  }
  return obj->update_inventory(item, delta);
}

InventoryQuantity ResourceManager::get_inventory(GridObjectId object_id, InventoryItem item) const {
  const HasInventory* obj = _get_inventory_object(object_id);
  if (!obj) {
    return 0;
  }
  const auto& inventory = obj->get_inventory();
  auto it = inventory.find(item);
  return (it != inventory.end()) ? it->second : 0;
}

const std::map<InventoryItem, InventoryQuantity>& ResourceManager::get_inventory(GridObjectId object_id) const {
  const HasInventory* obj = _get_inventory_object(object_id);
  if (!obj) {
    static const std::map<InventoryItem, InventoryQuantity> empty_inventory;
    return empty_inventory;
  }
  return obj->get_inventory();
}

// Utility methods
std::vector<GridObjectId> ResourceManager::get_all_inventory_object_ids() const {
  std::vector<GridObjectId> ids;
  for (const auto& [_, objects] : _objects_by_group) {
    for (const auto* obj : objects) {
      // We need to cast to GridObject to get the id
      const GridObject* grid_obj = dynamic_cast<const GridObject*>(obj);
      if (grid_obj) {
        ids.push_back(grid_obj->id);
      }
    }
  }
  return ids;
}

bool ResourceManager::is_inventory_object(GridObjectId id) const {
  return _get_inventory_object(id) != nullptr;
}

// Resource transfer between objects
InventoryDelta ResourceManager::transfer_resource(GridObjectId from_id, GridObjectId to_id, InventoryItem item, InventoryQuantity amount) {
  HasInventory* from_obj = _get_inventory_object(from_id);
  HasInventory* to_obj = _get_inventory_object(to_id);

  if (!from_obj || !to_obj) {
    return 0;
  }

  // Check if source has enough resources
  InventoryQuantity available = get_inventory(from_id, item);
  if (available < amount) {
    amount = available;  // Transfer what's available
  }

  if (amount == 0) {
    return 0;
  }

  // Remove from source
  from_obj->update_inventory(item, -static_cast<InventoryDelta>(amount));

  // Add to destination
  InventoryDelta added = to_obj->update_inventory(item, static_cast<InventoryDelta>(amount));

  return added;  // Return actual amount transferred
}

// Group-based resource operations
InventoryQuantity ResourceManager::get_group_total_inventory(const std::string& group_name, InventoryItem item) const {
  auto it = _group_inventory_totals.find(group_name);
  if (it == _group_inventory_totals.end()) {
    return 0;
  }

  const auto& group_totals = it->second;
  auto item_it = group_totals.find(item);
  return (item_it != group_totals.end()) ? item_it->second : 0;
}

void ResourceManager::distribute_resources_to_group(const std::string& group_name, InventoryItem item, InventoryQuantity total_amount) {
  const auto& objects = get_objects_by_group(group_name);
  if (objects.empty()) {
    return;
  }

  // Simple equal distribution
  InventoryQuantity per_object = total_amount / objects.size();
  InventoryQuantity remainder = total_amount % objects.size();

  for (size_t i = 0; i < objects.size(); ++i) {
    InventoryQuantity amount = per_object + (i < remainder ? 1 : 0);
    objects[i]->update_inventory(item, static_cast<InventoryDelta>(amount));
  }
}

// Weighted random selection methods
GridObjectId ResourceManager::select_random_object_by_resource(const std::string& group_name, InventoryItem item) const {
  const auto& objects = get_objects_by_group(group_name);
  if (objects.empty()) {
    return 0;  // Invalid ID
  }

  // Calculate total weight (total quantity of the item in the group)
  InventoryQuantity total_weight = 0;
  for (const auto* obj : objects) {
    total_weight += get_inventory(dynamic_cast<const GridObject*>(obj)->id, item);
  }

  if (total_weight == 0) {
    // No objects have this resource, return random object
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, objects.size() - 1);
    return dynamic_cast<const GridObject*>(objects[dis(gen)])->id;
  }

  // Weighted random selection
  static std::random_device rd;
  static std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(1, total_weight);
  InventoryQuantity target_weight = dis(gen);

  InventoryQuantity current_weight = 0;
  for (const auto* obj : objects) {
    current_weight += get_inventory(dynamic_cast<const GridObject*>(obj)->id, item);
    if (current_weight >= target_weight) {
      return dynamic_cast<const GridObject*>(obj)->id;
    }
  }

  // Fallback (shouldn't happen)
  return dynamic_cast<const GridObject*>(objects.back())->id;
}

// Inventory change callback
void ResourceManager::on_inventory_changed(GridObjectId object_id, InventoryItem item, InventoryDelta delta) {
  HasInventory* obj = _get_inventory_object(object_id);
  if (!obj) {
    return;
  }

  std::string group_name = _get_group_name(obj);

  // Update group inventory totals
  auto& group_totals = _group_inventory_totals[group_name];
  group_totals[item] += delta;

  // Remove zero entries
  if (group_totals[item] == 0) {
    group_totals.erase(item);
  }

  // Remove empty group entries
  if (group_totals.empty()) {
    _group_inventory_totals.erase(group_name);
  }
}

// Private helper methods
HasInventory* ResourceManager::_get_inventory_object(GridObjectId object_id) const {
  GridObject* grid_obj = _grid->object(object_id);
  if (!grid_obj) {
    return nullptr;
  }

  return dynamic_cast<HasInventory*>(grid_obj);
}

std::string ResourceManager::_get_group_name(HasInventory* obj) const {
  // Try to cast to Agent first to get group_name
  Agent* agent = dynamic_cast<Agent*>(obj);
  if (agent) {
    return agent->group_name;
  }

  // For converters and other objects, use empty string
  return "";
}

void ResourceManager::_validate_object_id(GridObjectId id) const {
  if (!_get_inventory_object(id)) {
    throw std::runtime_error("Invalid object ID: " + std::to_string(id));
  }
}

void ResourceManager::_update_group_maps() {
  _objects_by_group.clear();

  // Iterate through all objects in the grid
  for (size_t i = 0; i < _grid->objects.size(); ++i) {
    GridObject* grid_obj = _grid->objects[i].get();
    HasInventory* inventory_obj = dynamic_cast<HasInventory*>(grid_obj);

    if (inventory_obj) {
      std::string group_name = _get_group_name(inventory_obj);
      _objects_by_group[group_name].push_back(inventory_obj);
    }
  }
}

void ResourceManager::_update_inventory_totals() {
  _group_inventory_totals.clear();

  // Recalculate all group inventory totals
  for (const auto& [group_name, objects] : _objects_by_group) {
    auto& group_totals = _group_inventory_totals[group_name];

    for (const auto* obj : objects) {
      const auto& inventory = obj->get_inventory();
      for (const auto& [item, quantity] : inventory) {
        group_totals[item] += quantity;
      }
    }
  }
}
