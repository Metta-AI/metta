#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_HAS_INVENTORY_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_HAS_INVENTORY_HPP_

#include <algorithm>
#include <string>
#include <unordered_map>

#include "objects/constants.hpp"
#include "objects/inventory.hpp"
#include "objects/inventory_config.hpp"
class HasInventory {
public:
  explicit HasInventory(const InventoryConfig& inventory_config) : inventory(inventory_config) {}
  virtual ~HasInventory() = default;
  Inventory inventory;

  // Whether the inventory is accessible to an agent.
  virtual bool inventory_is_accessible() {
    return true;
  }

  virtual InventoryDelta update_inventory(InventoryItem item, InventoryDelta delta) {
    return inventory.update(item, delta);
  }
};

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_HAS_INVENTORY_HPP_
