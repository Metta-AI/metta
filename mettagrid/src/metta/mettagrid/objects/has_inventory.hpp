#ifndef OBJECTS_HAS_INVENTORY_HPP_
#define OBJECTS_HAS_INVENTORY_HPP_

#include <algorithm>
#include <map>
#include <string>

#include "constants.hpp"
#include "metta_object.hpp"

class HasInventory : public MettaObject {
public:
  vector<unsigned char> inventory;

  void init_has_inventory(ObjectConfig cfg) {
    this->inventory.resize(InventoryItem::InventoryItemCount);
  }

  // Whether the inventory is accessible to an agent.
  virtual bool inventory_is_accessible() {
    return true;
  }

  virtual int update_inventory(InventoryItem item, short amount) {
    int initial_amount = this->inventory[item];
    int new_amount = initial_amount + amount;
    new_amount = std::clamp(new_amount, 0, 255);
    this->inventory[item] = new_amount;
    return new_amount - initial_amount;
  }
};

#endif  // OBJECTS_HAS_INVENTORY_HPP_
