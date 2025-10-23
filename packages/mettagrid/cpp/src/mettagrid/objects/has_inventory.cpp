#include "objects/has_inventory.hpp"

#include <algorithm>
#include <cassert>

// Static method for shared update
InventoryDelta HasInventory::shared_update(std::vector<HasInventory*> inventory_havers,
                                           InventoryItem item,
                                           InventoryDelta delta) {
  if (inventory_havers.empty()) {
    return 0;
  }
  // We expect the main usage to be 3 passes:
  // 1. Separate inventories into those that can fully participate and those that can't. During this stage we
  // update the not-fully-participating inventories as much as possible.
  // 2. Confirm that all remaining inventories can fully participate, based on an updated understanding of what's
  // needed (the per-inventory amount, since some inventories may have dropped out).
  // 3. Update all inventories that can fully participate.
  //
  // One thing we specifically aim for is that the earlier inventories get more of the update, if it can't be
  // evenly split.
  InventoryDelta delta_remaining = delta;
  std::vector<HasInventory*> inventory_havers_to_consider;
  std::vector<HasInventory*> next_inventory_havers_to_consider = inventory_havers;
  // We want this to be a signed type, since otherwise we'll have a signed division issue.
  int num_inventory_havers_remaining = next_inventory_havers_to_consider.size();
  // Intentionally rounded towards zero.
  InventoryDelta delta_per_inventory_haver = delta_remaining / num_inventory_havers_remaining;
  do {
    inventory_havers_to_consider = next_inventory_havers_to_consider;
    next_inventory_havers_to_consider.clear();
    for (HasInventory* inventory_haver : inventory_havers_to_consider) {
      // Check to see if we have enough information to update this inventory now. I.e., do we know that this update
      // is going to fill / empty the inventory, so we can just do that?
      bool update_immediately;
      if (delta_remaining > 0) {
        update_immediately = inventory_haver->inventory.free_space(item) <= delta_per_inventory_haver;
      } else {
        // For negative delta, update immediately if inventory has less than or equal to what we want to take
        update_immediately = inventory_haver->inventory.amount(item) <= -delta_per_inventory_haver;
      }
      if (update_immediately) {
        // Update the inventory by as much as we can, and adjust how much we have left.
        delta_remaining -= inventory_haver->update_inventory(item, delta_per_inventory_haver);
        num_inventory_havers_remaining--;
        if (num_inventory_havers_remaining > 0) {
          delta_per_inventory_haver = delta_remaining / num_inventory_havers_remaining;
        }
      } else {
        next_inventory_havers_to_consider.push_back(inventory_haver);
      }
    }
    // Do this until we don't kick any inventories off the list. Once we're here, all remaining inventories can
    // "fully participate".
  } while (inventory_havers_to_consider.size() != next_inventory_havers_to_consider.size());

  if (num_inventory_havers_remaining == 0) {
    return delta - delta_remaining;
  }

  // Update in reverse order. Because of the direction of rounding, this means that the earlier inventories will get
  // more of the delta (if it's not evenly split).
  for (int i = inventory_havers_to_consider.size() - 1; i >= 0; i--) {
    HasInventory* inventory_haver = inventory_havers_to_consider[i];
    InventoryDelta inventory_delta = delta_remaining / (i + 1);
    InventoryDelta actual_delta = inventory_haver->update_inventory(item, inventory_delta);
    assert(actual_delta == inventory_delta && "Expected inventory_haver to absorb all of the delta");
    delta_remaining -= actual_delta;
  }
  assert(delta_remaining == 0 && "Expected all of the delta to be consumed");

  return delta - delta_remaining;
}

// Whether the inventory is accessible to an agent.
bool HasInventory::inventory_is_accessible() {
  return true;
}

// Update the inventory for a specific item
InventoryDelta HasInventory::update_inventory(InventoryItem item, InventoryDelta delta) {
  return inventory.update(item, delta);
}
