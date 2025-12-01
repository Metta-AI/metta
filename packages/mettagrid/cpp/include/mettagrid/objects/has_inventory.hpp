#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_HAS_INVENTORY_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_HAS_INVENTORY_HPP_

#include <string>
#include <unordered_map>
#include <vector>

#include "objects/constants.hpp"
#include "objects/inventory.hpp"
#include "objects/inventory_config.hpp"

class HasInventory {
public:
  explicit HasInventory(const InventoryConfig& inventory_config,
                        const std::vector<std::string>* resource_names = nullptr,
                        const std::unordered_map<std::string, ObservationType>* feature_ids = nullptr)
      : inventory(inventory_config, resource_names, feature_ids) {}
  virtual ~HasInventory() = default;
  Inventory inventory;

  // Splits the delta between the inventories. Returns the amount of delta successfully consumed.
  static InventoryDelta shared_update(std::vector<HasInventory*> inventory_havers,
                                      InventoryItem item,
                                      InventoryDelta delta);

  // Transfer resources from source to target. Returns the amount actually transferred.
  // If destroy_untransferred_resources is true, the source loses min(delta, available) resources
  // even if the target cannot accept all of them.
  static InventoryDelta transfer_resources(HasInventory& source,
                                           HasInventory& target,
                                           InventoryItem item,
                                           InventoryDelta delta,
                                           bool destroy_untransferred_resources);

  // Whether the inventory is accessible to an agent.
  virtual bool inventory_is_accessible();

  // Update the inventory for a specific item
  virtual InventoryDelta update_inventory(InventoryItem item, InventoryDelta delta);
};

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_HAS_INVENTORY_HPP_
