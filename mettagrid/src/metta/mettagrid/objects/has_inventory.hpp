#ifndef OBJECTS_HAS_INVENTORY_HPP_
#define OBJECTS_HAS_INVENTORY_HPP_

#include <algorithm>
#include <map>
#include <string>

#include "constants.hpp"
#include "grid_object.hpp"
#include "inventory_list.hpp"

class HasInventory : public GridObject {
public:
  // Whether the inventory is accessible to an agent.
  virtual bool inventory_is_accessible() const {
    return true;
  }

  // Get the inventory list - must be implemented by derived classes
  virtual InventoryList& get_inventory_list() = 0;
  virtual const InventoryList& get_inventory_list() const = 0;


  virtual InventoryDelta update_inventory(InventoryItem item, InventoryDelta delta) {
    // Default implementation delegates to InventoryList for advanced resource tracking
    // Derived classes can override if they need special behavior
    InventoryList& inventory_list = get_inventory_list();

    // Use empty resource loss probability map for basic inventory management
    // Derived classes can override to provide their own resource loss probabilities
    std::map<InventoryItem, float> empty_resource_loss_prob;

    return inventory_list.update_inventory(item, delta, empty_resource_loss_prob, this->id);
  }
};

#endif  // OBJECTS_HAS_INVENTORY_HPP_
