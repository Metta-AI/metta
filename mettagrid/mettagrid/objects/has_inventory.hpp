#ifndef HAS_INVENTORY_HPP
#define HAS_INVENTORY_HPP

#include <algorithm>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "constants.hpp"
#include "objects/metta_object.hpp"

class HasInventory : public MettaObject {
public:
  std::vector<uint8_t> inventory;

  void init_has_inventory(ObjectConfig cfg) {
    this->inventory.resize(InventoryItem::InventoryCount);
  }

  virtual bool has_inventory() override {
    return true;
  }

  // Whether the inventory is accessible to an agent.
  virtual bool inventory_is_accessible() {
    return true;
  }

  virtual void update_inventory(InventoryItem item, int16_t amount) {
    int32_t current = this->inventory[item];
    int32_t new_value = current + amount;

    // Clamp the result between 0 and UINT8_MAX
    this->inventory[item] = static_cast<uint8_t>(std::clamp(new_value, 0, static_cast<int32_t>(UINT8_MAX)));
  }

  virtual void obs(ObsType* obs) const override {
    MettaObject::obs(obs);

    // HasInventory-specific features
    encode(obs, GridFeature::HAS_INVENTORY, 1);

    // We don't encode inventory here because different derived classes
    // may want to encode inventory with different prefixes
  }
};

#endif