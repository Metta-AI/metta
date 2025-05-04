#ifndef HAS_INVENTORY_HPP
#define HAS_INVENTORY_HPP

#include <map>
#include <string>

#include "constants.hpp"
#include "metta_object.hpp"

class HasInventory : public MettaObject {
public:
  vector<unsigned char> inventory;

  void init_has_inventory(ObjectConfig cfg) {
    this->inventory.resize(InventoryItem::InventoryCount);
  }

  virtual bool has_inventory() {
    return true;
  }

  // Whether the inventory is accessible to an agent.
  virtual bool inventory_is_accessible() {
    return true;
  }

  virtual void update_inventory(InventoryItem item, short amount) {
    if (amount + this->inventory[item] > 255) {
      amount = 255 - this->inventory[item];
    }
    if (amount + this->inventory[item] < 0) {
      amount = -this->inventory[item];
    }
    this->inventory[item] += amount;
  }
};

#endif
