#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_INVENTORY_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_INVENTORY_HPP_

#include <algorithm>
#include <map>
#include <string>

#include "core/types.hpp"
#include "objects/constants.hpp"

class Inventory {
private:
  std::map<InventoryItem, InventoryQuantity> _inventory;

public:
  Inventory() : _inventory() {}

  InventoryDelta update(InventoryItem item, InventoryDelta attempted_delta) {
    InventoryQuantity initial_amount = this->_inventory[item];
    int new_amount = static_cast<int>(initial_amount + attempted_delta);

    constexpr InventoryQuantity min = std::numeric_limits<InventoryQuantity>::min();
    InventoryQuantity max = std::numeric_limits<InventoryQuantity>::max();

    InventoryQuantity clamped_amount = static_cast<InventoryQuantity>(std::clamp<int>(new_amount, min, max));

    if (clamped_amount == 0) {
      this->_inventory.erase(item);
    } else {
      this->_inventory[item] = clamped_amount;
    }

    InventoryDelta clamped_delta = clamped_amount - initial_amount;
    return clamped_delta;
  }

  InventoryQuantity amount(InventoryItem item) const {
    if (this->_inventory.count(item) == 0) {
      return 0;
    }
    return this->_inventory.at(item);
  }

  std::map<InventoryItem, InventoryQuantity> get() const {
    return this->_inventory;
  }
};

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_INVENTORY_HPP_
