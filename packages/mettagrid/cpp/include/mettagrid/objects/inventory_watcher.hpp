#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_INVENTORY_WATCHER_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_INVENTORY_WATCHER_HPP_

#include "core/types.hpp"

class Inventory;

class InventoryWatcher {
public:
  virtual void onInventoryChange(Inventory& inventory) {}
  virtual void onInventoryChange(Inventory& inventory, InventoryItem item, InventoryDelta delta) {
    this->onInventoryChange(inventory);
  }
};

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_INVENTORY_WATCHER_HPP_
