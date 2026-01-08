#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_FACTION_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_FACTION_HPP_

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/types.hpp"
#include "objects/faction_config.hpp"
#include "objects/has_inventory.hpp"
#include "objects/inventory.hpp"
#include "systems/stats_tracker.hpp"

// Forward declaration
class Alignable;
class GridObject;

class Faction : public HasInventory {
private:
  std::vector<Alignable*> _members;
  // Track count of aligned objects by type name
  std::unordered_map<std::string, int> _aligned_counts;

public:
  std::string name;
  StatsTracker stats;

  explicit Faction(const FactionConfig& cfg,
                   const std::vector<std::string>* resource_names = nullptr,
                   const std::unordered_map<std::string, ObservationType>* feature_ids = nullptr)
      : HasInventory(cfg.inventory_config, resource_names, feature_ids), name(cfg.name), stats(resource_names) {
    // Set initial inventory (ignore limits for initial setup)
    for (const auto& [resource, amount] : cfg.initial_inventory) {
      if (amount > 0) {
        inventory.update(resource, amount, /*ignore_limits=*/true);
      }
    }
  }

  virtual ~Faction() = default;

  // Add a member to this faction (with optional type name for stats tracking)
  void addMember(Alignable* obj, const std::string& type_name = "") {
    if (obj && std::find(_members.begin(), _members.end(), obj) == _members.end()) {
      _members.push_back(obj);
      if (!type_name.empty()) {
        _aligned_counts[type_name]++;
        stats.set("aligned." + type_name, static_cast<float>(_aligned_counts[type_name]));
      }
    }
  }

  // Remove a member from this faction (with optional type name for stats tracking)
  void removeMember(Alignable* obj, const std::string& type_name = "") {
    auto it = std::find(_members.begin(), _members.end(), obj);
    if (it != _members.end()) {
      _members.erase(it);
      if (!type_name.empty() && _aligned_counts.count(type_name) > 0) {
        _aligned_counts[type_name]--;
        stats.set("aligned." + type_name, static_cast<float>(_aligned_counts[type_name]));
      }
    }
  }

  // Get all members
  const std::vector<Alignable*>& members() const {
    return _members;
  }

  // Get the number of members
  size_t memberCount() const {
    return _members.size();
  }

  // Get aligned counts by type (for held duration tracking)
  const std::unordered_map<std::string, int>& aligned_counts() const {
    return _aligned_counts;
  }

  // Update held duration stats (call once per tick)
  void update_held_stats() {
    for (const auto& [type_name, count] : _aligned_counts) {
      if (count > 0) {
        stats.add("aligned." + type_name + ".held", static_cast<float>(count));
      }
    }
  }

  // Update inventory stats (call at end of episode or as needed)
  void update_inventory_stats() {
    for (const auto& [item, quantity] : inventory.get()) {
      stats.set("inventory." + stats.resource_name(item), static_cast<float>(quantity));
    }
  }
};

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_FACTION_HPP_
