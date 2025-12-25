#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_COMMONS_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_COMMONS_HPP_

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/types.hpp"
#include "objects/commons_config.hpp"
#include "objects/has_inventory.hpp"
#include "objects/inventory.hpp"

// Forward declaration
class GridObject;

class Commons : public HasInventory {
private:
  std::vector<GridObject*> _members;

public:
  std::string name;

  explicit Commons(const CommonsConfig& cfg,
                   const std::vector<std::string>* resource_names = nullptr,
                   const std::unordered_map<std::string, ObservationType>* feature_ids = nullptr)
      : HasInventory(cfg.inventory_config, resource_names, feature_ids), name(cfg.name) {
    // Set initial inventory (ignore limits for initial setup)
    for (const auto& [resource, amount] : cfg.initial_inventory) {
      if (amount > 0) {
        inventory.update(resource, amount, /*ignore_limits=*/true);
      }
    }
  }

  virtual ~Commons() = default;

  // Add a member to this commons
  void addMember(GridObject* obj) {
    if (obj && std::find(_members.begin(), _members.end(), obj) == _members.end()) {
      _members.push_back(obj);
    }
  }

  // Remove a member from this commons
  void removeMember(GridObject* obj) {
    auto it = std::find(_members.begin(), _members.end(), obj);
    if (it != _members.end()) {
      _members.erase(it);
    }
  }

  // Get all members
  const std::vector<GridObject*>& members() const {
    return _members;
  }

  // Get the number of members
  size_t memberCount() const {
    return _members.size();
  }
};

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_COMMONS_HPP_
