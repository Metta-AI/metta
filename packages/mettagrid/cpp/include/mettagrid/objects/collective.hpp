#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_COLLECTIVE_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_COLLECTIVE_HPP_

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

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

  // Add a member to this collective
  void addMember(Alignable* obj) {
    if (obj && std::find(_members.begin(), _members.end(), obj) == _members.end()) {
      _members.push_back(obj);
    }
  }

  // Remove a member from this collective
  void removeMember(Alignable* obj) {
    auto it = std::find(_members.begin(), _members.end(), obj);
    if (it != _members.end()) {
      _members.erase(it);
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

  // Track stats when inventory changes
  void on_inventory_change(InventoryItem item, InventoryDelta delta) override {
    if (delta == 0) return;
    // We may want to track these by collective name, but that would make them clunkier for calculating reward.
    if (delta > 0) {
      stats.add("collective." + stats.resource_name(item) + ".deposited", delta);
    } else {
      stats.add("collective." + stats.resource_name(item) + ".withdrawn", -delta);
    }
  }
};

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_COLLECTIVE_HPP_
