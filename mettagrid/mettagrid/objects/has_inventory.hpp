#ifndef HAS_INVENTORY_HPP
#define HAS_INVENTORY_HPP

#include <cstdint>  // Added for standard integer types
#include <map>
#include <string>
#include <vector>

#include "objects/constants.hpp"
#include "objects/metta_object.hpp"

class HasInventory : public MettaObject {
public:
  std::vector<uint8_t> inventory;  // Changed from unsigned char to uint8_t

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

  virtual void update_inventory(InventoryItem item, int16_t amount) {
    if (amount + this->inventory[item] > UINT8_MAX) {
      amount = UINT8_MAX - this->inventory[item];
    }
    if (amount + this->inventory[item] < 0) {
      amount = -this->inventory[item];
    }
    this->inventory[item] += amount;
  }
};

#endif