#ifndef OBJECTS_HAS_INVENTORY_HPP_
#define OBJECTS_HAS_INVENTORY_HPP_

#include <algorithm>
#include <map>
#include <string>

#include "constants.hpp"

class HasInventory : public GridObject {
public:
  std::map<InventoryItem, InventoryQuantity> inventory;

  // Whether the inventory is accessible to an agent.
  virtual bool inventory_is_accessible() {
    return true;
  }

  virtual InventoryDelta update_inventory(InventoryItem item, InventoryDelta delta) {
    InventoryQuantity initial_amount = this->inventory[item];
    int new_amount = static_cast<int>(initial_amount + delta);

    constexpr int min = std::numeric_limits<InventoryQuantity>::min();
    constexpr int max = std::numeric_limits<InventoryQuantity>::max();
    InventoryQuantity clamped_amount = static_cast<InventoryQuantity>(std::clamp(new_amount, min, max));

    if (clamped_amount == 0) {
      this->inventory.erase(item);
    } else {
      this->inventory[item] = clamped_amount;
    }

    InventoryDelta clamped_delta = clamped_amount - initial_amount;
    return clamped_delta;
  }
};

#endif  // OBJECTS_HAS_INVENTORY_HPP_
