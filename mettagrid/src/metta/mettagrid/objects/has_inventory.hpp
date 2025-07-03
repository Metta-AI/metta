#ifndef OBJECTS_HAS_INVENTORY_HPP_
#define OBJECTS_HAS_INVENTORY_HPP_

#include <algorithm>
#include <map>
#include <string>

#include "constants.hpp"
#include "metta_object.hpp"

class HasInventory : public MettaObject {
public:
  std::map<InventoryItem, uint8_t> inventory;

  // Whether the inventory is accessible to an agent.
  virtual bool inventory_is_accessible() {
    return true;
  }

  virtual int update_inventory(InventoryItem item, short delta_amount) {
    int initial_amount = this->inventory[item];
    // Convert to int to handle negative amounts properly
    int new_amount = initial_amount + delta_amount;

    // Clamp to valid uint8_t range
    uint8_t clamped_amount = std::clamp(new_amount, 0, 255);

    if (clamped_amount == 0) {
      this->inventory.erase(item);
    } else {
      this->inventory[item] = clamped_amount;
    }
    return static_cast<int>(clamped_amount) - initial_amount;  // actual change
  }
};

#endif  // OBJECTS_HAS_INVENTORY_HPP_
