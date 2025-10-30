#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_INVENTORY_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_INVENTORY_HPP_

#include <string>
#include <unordered_map>
#include <vector>

#include "core/types.hpp"
#include "objects/constants.hpp"
#include "objects/inventory_config.hpp"

struct SharedInventoryLimit {
  InventoryQuantity limit;
  // How much do we have of whatever-this-limit-applies-to
  InventoryQuantity amount;
};

class Inventory {
private:
  std::unordered_map<InventoryItem, InventoryQuantity> _inventory;
  std::unordered_map<InventoryItem, SharedInventoryLimit*> _limits;
  std::vector<ObservationType> _item_feature_ids;  // Maps item index to observation feature ID

public:
  // Constructor and Destructor
  explicit Inventory(const InventoryConfig& cfg,
                     const std::vector<std::string>* resource_names = nullptr,
                     const std::unordered_map<std::string, ObservationType>* feature_ids = nullptr);
  ~Inventory();

  // Update the inventory for a specific item
  InventoryDelta update(InventoryItem item, InventoryDelta attempted_delta);

  // Get the amount of a specific item
  InventoryQuantity amount(InventoryItem item) const;

  // Get the free space for a specific item
  InventoryQuantity free_space(InventoryItem item) const;

  // Get all inventory items
  std::unordered_map<InventoryItem, InventoryQuantity> get() const;

  // Get the observation feature ID for an inventory item
  ObservationType get_feature_id(InventoryItem item) const;
};

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_INVENTORY_HPP_
