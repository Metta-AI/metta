#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_INVENTORY_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_INVENTORY_HPP_

#include <algorithm>
#include <map>
#include <string>

#include "core/types.hpp"
#include "objects/constants.hpp"
#include "systems/stats_tracker.hpp"

struct SharedInventoryLimit {
  InventoryQuantity limit;
  InventoryQuantity amount;
};

struct InventoryConfig {
  std::vector<std::vector<InventoryItem>, InventoryQuantity> limits;
};

class Inventory {
private:
  std::map<InventoryItem, InventoryQuantity> _inventory;
  std::map<InventoryItem, SharedInventoryLimit*> _limits;
  StatsTracker* _stats;

public:
  Inventory(const InventoryConfig& cfg) : _inventory(), _limits(), _stats(nullptr) {
    for (const auto& [resources, limit_value] : cfg.limits) {
      SharedInventoryLimit* limit = new SharedInventoryLimit();
      limit->amount = 0;
      limit->limit = limit_value;
      for (const auto& resource : resources) {
        this->_limits[resource] = limit;
      }
    }
  }

  InventoryDelta update(InventoryItem item, InventoryDelta attempted_delta) {
    InventoryQuantity initial_amount = this->_inventory[item];
    int new_amount = static_cast<int>(initial_amount + attempted_delta);

    constexpr InventoryQuantity min = std::numeric_limits<InventoryQuantity>::min();
    SharedInventoryLimit* limit = nullptr;
    if (this->_limits.count(item) > 0) {
      limit = this->_limits[item];
    }
    InventoryQuantity max = std::numeric_limits<InventoryQuantity>::max();
    if (limit != nullptr) {
      max = limit->limit;
    }

    InventoryQuantity clamped_amount = static_cast<InventoryQuantity>(std::clamp(new_amount, min, max));

    if (clamped_amount == 0) {
      this->_inventory.erase(item);
    } else {
      this->_inventory[item] = clamped_amount;
    }

    InventoryDelta clamped_delta = clamped_amount - initial_amount;
    return clamped_delta;
  }

  InventoryQuantity amount(InventoryItem item) const {
    if (this->_inventory.count(item) == 0) {
      return 0;
    }
    return this->_inventory[item];
  }

  std::map<InventoryItem, InventoryQuantity> get() const {
    return this->_inventory;
  }
};

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_INVENTORY_HPP_
