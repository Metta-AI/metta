#include "objects/has_inventory.hpp"

// Default implementation for inventory_is_accessible
bool HasInventory::inventory_is_accessible() {
  return true;  // Default: inventory is accessible
}

// Default implementation for update_inventory
InventoryDelta HasInventory::update_inventory(InventoryItem item, InventoryDelta delta) {
  return inventory.update(item, delta);  // Default: just update the inventory
}
