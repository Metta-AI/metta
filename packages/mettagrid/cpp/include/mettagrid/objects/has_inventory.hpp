#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_HAS_INVENTORY_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_HAS_INVENTORY_HPP_

#include <vector>

#include "objects/constants.hpp"
#include "objects/inventory.hpp"
#include "objects/inventory_config.hpp"

class HasInventory {
public:
  explicit HasInventory(const InventoryConfig& inventory_config) : inventory(inventory_config, this) {}
  virtual ~HasInventory() = default;
  Inventory inventory;

  // Callback method called when inventory changes
  // Override this method in derived classes to react to inventory changes
  virtual void on_inventory_change(InventoryItem item, InventoryDelta delta) {}

  // Splits the delta between the inventories. Returns the amount of delta successfully consumed.
  static InventoryDelta shared_update(std::vector<Inventory*> inventories, InventoryItem item, InventoryDelta delta);

  // Transfer resources from source to target. Returns the amount actually transferred.
  // If destroy_untransferred_resources is true, the source loses min(delta, available) resources
  // even if the target cannot accept all of them.
  static InventoryDelta transfer_resources(Inventory& source,
                                           Inventory& target,
                                           InventoryItem item,
                                           InventoryDelta delta,
                                           bool destroy_untransferred_resources);

  // Whether the inventory is accessible to an agent.
  virtual bool inventory_is_accessible();
};

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_HAS_INVENTORY_HPP_
