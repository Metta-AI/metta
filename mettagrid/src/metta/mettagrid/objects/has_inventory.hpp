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


  // Pure virtual method - must be implemented by derived classes
  virtual InventoryDelta update_inventory(InventoryItem item, InventoryDelta delta) = 0;
};

#endif  // OBJECTS_HAS_INVENTORY_HPP_
