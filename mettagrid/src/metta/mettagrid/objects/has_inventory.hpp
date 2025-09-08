#ifndef OBJECTS_HAS_INVENTORY_HPP_
#define OBJECTS_HAS_INVENTORY_HPP_

#include <algorithm>
#include <functional>
#include <map>
#include <string>

#include "constants.hpp"
#include "grid_object.hpp"

class HasInventory : public GridObject {
public:
  std::map<InventoryItem, InventoryQuantity> inventory;
  
  // Callback function type for inventory changes
  using InventoryChangeCallback = std::function<void(GridObjectId, InventoryItem, InventoryDelta)>;

  // Whether the inventory is accessible to an agent.
  virtual bool inventory_is_accessible() {
    return true;
  }
  
  // Set callback for inventory changes
  void set_inventory_callback(InventoryChangeCallback callback) {
    inventory_callback = callback;
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
    
    // Call callback if inventory actually changed
    if (clamped_delta != 0 && inventory_callback) {
      inventory_callback(this->id, item, clamped_delta);
    }
    
    return clamped_delta;
  }

private:
  InventoryChangeCallback inventory_callback;
};

#endif  // OBJECTS_HAS_INVENTORY_HPP_
