#include "resource_manager.hpp"

#include <algorithm>
#include <cassert>
#include <random>
#include <stdexcept>

#include "objects/agent.hpp"
#include "objects/converter.hpp"

ResourceManager::ResourceManager(Grid* grid, std::mt19937& rng) : _grid(grid), _rng(rng) {
  assert(_grid != nullptr);
  // Initialize group maps with existing objects
  _update_group_maps();
  // Initialize inventory totals
  _update_inventory_totals();
  // Initialize bins
  _update_bins();
}

void ResourceManager::step() {
  // Process resource loss for each cached bin - no iteration needed!
  
  // Collect updates to apply after iteration to avoid iterator invalidation
  std::vector<std::tuple<HasInventory*, InventoryItem, InventoryDelta>> updates;

  for (auto& [bin_key, objects_with_quantities] : _bins) {
    const InventoryItem& item = bin_key.second;

    if (objects_with_quantities.empty()) {
      continue;
    }

    // Get cached loss probability for this bin
    auto loss_it = _bin_loss_probabilities.find(bin_key);
    if (loss_it == _bin_loss_probabilities.end() || loss_it->second <= 0.0f) {
      continue;  // No loss probability for this item
    }

    float loss_probability = loss_it->second;

    // Calculate total resources in this bin
    InventoryQuantity total_resources = 0;
    for (const auto& [obj, quantity] : objects_with_quantities) {
      total_resources += quantity;
    }

    if (total_resources == 0) {
      continue;
    }

    // Use Bernoulli distribution for each resource (coin flip)
    InventoryQuantity resources_to_lose = 0;
    std::bernoulli_distribution bernoulli_dist(loss_probability);
    
    for (InventoryQuantity i = 0; i < total_resources; ++i) {
      if (bernoulli_dist(_rng)) {
        resources_to_lose++;
      }
    }

    if (resources_to_lose == 0) {
      continue;
    }

    // Randomly assign losses to individual objects using uniform distribution
    if (objects_with_quantities.empty()) {
      continue;  // Safety check
    }
    
    std::vector<InventoryQuantity> losses_per_object(objects_with_quantities.size(), 0);

    for (InventoryQuantity i = 0; i < resources_to_lose; ++i) {
      // Uniformly select an object to lose a resource
      std::uniform_int_distribution<> obj_dis(0, objects_with_quantities.size() - 1);
      size_t obj_index = obj_dis(_rng);

      // Check if this object still has resources to lose
      if (losses_per_object[obj_index] < objects_with_quantities[obj_index].second) {
        losses_per_object[obj_index]++;
      }
    }

    // Collect the losses to apply later
    for (size_t i = 0; i < objects_with_quantities.size(); ++i) {
      if (losses_per_object[i] > 0) {
        HasInventory* obj = objects_with_quantities[i].first;
        InventoryDelta decrease_amount = -static_cast<InventoryDelta>(losses_per_object[i]);
        updates.push_back({obj, item, decrease_amount});
      }
    }
  }
  
  // Apply all updates after iteration to avoid iterator invalidation
  for (const auto& [obj, item, delta] : updates) {
    obj->update_inventory(item, delta);
  }
}

// Object registration and tracking methods
void ResourceManager::register_inventory_object(HasInventory* obj, const std::string& group_name) {
  if (!obj) return;

  GridObject* grid_obj = dynamic_cast<GridObject*>(obj);
  if (!grid_obj) return;

  std::string full_group_name;
  if (group_name.empty()) {
    // Use the cached group name or construct from type
    full_group_name = _get_group_name(obj);
  } else {
    // Use the provided group name with type prefix
    full_group_name = obj->type_name() + ":" + group_name;
    // Cache the group name
    _object_group_names[grid_obj->id] = full_group_name;
  }

  _objects_by_group[full_group_name].push_back(obj);

  // Add object to bins
  _add_object_to_bins(obj, full_group_name);
}

void ResourceManager::register_agent(HasInventory* obj, const std::string& group_name) {
  register_inventory_object(obj, group_name);
}

void ResourceManager::register_object(HasInventory* obj) {
  register_inventory_object(obj, "");
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

  // Clean up cached group name
  _object_group_names.erase(object_id);

  // Remove object from bins
  _remove_object_from_bins(object_id, group_name);
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

  // Update bins
  _update_object_in_bins(obj, group_name, item, delta);
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
  // Cast to GridObject to get the ID
  GridObject* grid_obj = dynamic_cast<GridObject*>(obj);
  if (!grid_obj) {
    return "unknown:";
  }

  // Look up the cached group name
  auto it = _object_group_names.find(grid_obj->id);
  if (it != _object_group_names.end()) {
    return it->second;
  }

  // Fallback: construct group name from type (shouldn't happen if registration is correct)
  return obj->type_name() + ":";
}

void ResourceManager::_validate_object_id(GridObjectId id) const {
  if (!_get_inventory_object(id)) {
    throw std::runtime_error("Invalid object ID: " + std::to_string(id));
  }
}

void ResourceManager::_update_group_maps() {
  _objects_by_group.clear();
  _object_group_names.clear();

  // Iterate through all objects in the grid
  for (size_t i = 0; i < _grid->objects.size(); ++i) {
    GridObject* grid_obj = _grid->objects[i].get();
    HasInventory* inventory_obj = dynamic_cast<HasInventory*>(grid_obj);

    if (inventory_obj) {
      // Use the generic registration method which will call _get_group_name
      // This is for initialization only - in practice, use specific registration methods
      register_inventory_object(inventory_obj);
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

void ResourceManager::_update_bins() {
  _bins.clear();
  _bin_loss_probabilities.clear();

  // Rebuild bins from scratch
  for (const auto& [group_name, objects] : _objects_by_group) {
    for (HasInventory* obj : objects) {
      _add_object_to_bins(obj, group_name);
    }
  }
}

void ResourceManager::_add_object_to_bins(HasInventory* obj, const std::string& group_name) {
  const auto& inventory = obj->get_inventory();
  const auto& loss_probs = obj->get_resource_loss_prob();

  for (const auto& [item, quantity] : inventory) {
    if (quantity > 0) {
      auto bin_key = std::make_pair(group_name, item);
      _bins[bin_key].push_back({obj, quantity});

      // Cache loss probability for this bin (from first object)
      if (_bin_loss_probabilities.find(bin_key) == _bin_loss_probabilities.end()) {
        auto loss_it = loss_probs.find(item);
        if (loss_it != loss_probs.end()) {
          _bin_loss_probabilities[bin_key] = loss_it->second;
        }
      }
    }
  }
}

void ResourceManager::_remove_object_from_bins(GridObjectId object_id, const std::string& group_name) {
  // Remove all entries for this object from all bins
  for (auto& [bin_key, objects_with_quantities] : _bins) {
    if (bin_key.first == group_name) {
      objects_with_quantities.erase(
        std::remove_if(objects_with_quantities.begin(), objects_with_quantities.end(),
          [object_id](const std::pair<HasInventory*, InventoryQuantity>& entry) {
            GridObject* grid_obj = dynamic_cast<GridObject*>(entry.first);
            return grid_obj && grid_obj->id == object_id;
          }),
        objects_with_quantities.end());

      // Remove empty bins
      if (objects_with_quantities.empty()) {
        _bins.erase(bin_key);
        _bin_loss_probabilities.erase(bin_key);
      }
    }
  }
}

void ResourceManager::_update_object_in_bins(HasInventory* obj, const std::string& group_name, InventoryItem item, InventoryDelta delta) {
  auto bin_key = std::make_pair(group_name, item);

  // Find the object in the bin and update its quantity
  auto& objects_with_quantities = _bins[bin_key];
  for (auto& [bin_obj, quantity] : objects_with_quantities) {
    if (bin_obj == obj) {
      quantity += delta;

      // Remove if quantity becomes zero or negative
      if (quantity <= 0) {
        objects_with_quantities.erase(
          std::remove(objects_with_quantities.begin(), objects_with_quantities.end(), std::make_pair(obj, quantity)),
          objects_with_quantities.end());

        // Remove empty bins
        if (objects_with_quantities.empty()) {
          _bins.erase(bin_key);
          _bin_loss_probabilities.erase(bin_key);
        }
      }
      return;
    }
  }

  // If object not found in bin and delta is positive, add it
  if (delta > 0) {
    objects_with_quantities.push_back({obj, static_cast<InventoryQuantity>(delta)});

    // Cache loss probability if not already cached
    if (_bin_loss_probabilities.find(bin_key) == _bin_loss_probabilities.end()) {
      const auto& loss_probs = obj->get_resource_loss_prob();
      auto loss_it = loss_probs.find(item);
      if (loss_it != loss_probs.end()) {
        _bin_loss_probabilities[bin_key] = loss_it->second;
      }
    }
  }
}
