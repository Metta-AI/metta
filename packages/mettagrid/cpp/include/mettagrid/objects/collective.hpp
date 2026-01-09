#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_COLLECTIVE_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_COLLECTIVE_HPP_

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/grid_object.hpp"
#include "core/types.hpp"
#include "objects/collective_config.hpp"
#include "objects/has_inventory.hpp"
#include "objects/inventory.hpp"
#include "systems/stats_tracker.hpp"

// Forward declaration
class Alignable;

class Collective : public HasInventory {
private:
  std::vector<Alignable*> _members;
  std::unordered_map<std::string, int> _aligned_counts;  // type_name -> count

  // Helper to get type_name from Alignable via dynamic_cast to GridObject
  static std::string get_type_name(Alignable* obj) {
    GridObject* grid_obj = dynamic_cast<GridObject*>(obj);
    return grid_obj ? grid_obj->type_name : "";
  }

public:
  std::string name;
  StatsTracker stats;

  explicit Collective(const CollectiveConfig& cfg, const std::vector<std::string>* resource_names)
      : HasInventory(cfg.inventory_config), name(cfg.name), stats(resource_names) {
    // Set initial inventory (ignore limits for initial setup)
    for (const auto& [resource, amount] : cfg.initial_inventory) {
      if (amount > 0) {
        inventory.update(resource, amount, /*ignore_limits=*/true);
      }
    }
  }

  virtual ~Collective() = default;

  // Add a member to this collective (type_name read from GridObject)
  void addMember(Alignable* obj) {
    if (obj && std::find(_members.begin(), _members.end(), obj) == _members.end()) {
      _members.push_back(obj);
      std::string type_name = get_type_name(obj);
      _aligned_counts[type_name]++;
      stats.set("aligned." + type_name, static_cast<float>(_aligned_counts[type_name]));
    }
  }

  // Remove a member from this collective (type_name read from GridObject)
  void removeMember(Alignable* obj) {
    auto it = std::find(_members.begin(), _members.end(), obj);
    if (it != _members.end()) {
      _members.erase(it);
      std::string type_name = get_type_name(obj);
      _aligned_counts[type_name]--;
      if (_aligned_counts[type_name] <= 0) {
        _aligned_counts.erase(type_name);
        stats.set("aligned." + type_name, 0.0f);
      } else {
        stats.set("aligned." + type_name, static_cast<float>(_aligned_counts[type_name]));
      }
    }
  }

  // Update held duration stats (call once per tick)
  void update_held_stats() {
    for (const auto& [type_name, count] : _aligned_counts) {
      stats.add("aligned." + type_name + ".held", static_cast<float>(count));
    }
  }

  // Get aligned count for a specific type
  int get_aligned_count(const std::string& type_name) const {
    auto it = _aligned_counts.find(type_name);
    return it != _aligned_counts.end() ? it->second : 0;
  }

  // Get all aligned counts
  const std::unordered_map<std::string, int>& aligned_counts() const {
    return _aligned_counts;
  }

  // Get all members
  const std::vector<Alignable*>& members() const {
    return _members;
  }

  // Get the number of members
  size_t memberCount() const {
    return _members.size();
  }

  // Track stats when inventory changes
  void on_inventory_change(InventoryItem item, InventoryDelta delta) override {
    if (delta == 0) return;
    if (delta > 0) {
      stats.add("collective." + stats.resource_name(item) + ".deposited", delta);
    } else {
      stats.add("collective." + stats.resource_name(item) + ".withdrawn", -delta);
    }
  }
};

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_COLLECTIVE_HPP_
