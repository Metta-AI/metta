#include "objects/has_inventory.hpp"

#include <algorithm>
#include <cassert>

// Static method for shared update
InventoryDelta HasInventory::shared_update(std::vector<Inventory*> inventories,
                                           InventoryItem item,
                                           InventoryDelta delta) {
  if (inventories.empty()) {
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
  std::vector<Inventory*> inventories_to_consider;
  std::vector<Inventory*> next_inventories_to_consider = inventories;
  // We want this to be a signed type, since otherwise we'll have a signed division issue.
  int num_inventories_remaining = next_inventories_to_consider.size();
  // Intentionally rounded towards zero.
  InventoryDelta delta_per_inventory = delta_remaining / num_inventories_remaining;
  do {
    inventories_to_consider = next_inventories_to_consider;
    next_inventories_to_consider.clear();
    for (Inventory* inventory : inventories_to_consider) {
      // Check to see if we have enough information to update this inventory now. I.e., do we know that this update
      // is going to fill / empty the inventory, so we can just do that?
      bool update_immediately;
      if (delta_remaining > 0) {
        update_immediately = inventory->free_space(item) <= delta_per_inventory;
      } else {
        // For negative delta, update immediately if inventory has less than or equal to what we want to take
        update_immediately = inventory->amount(item) <= -delta_per_inventory;
      }
      if (update_immediately) {
        // Update the inventory by as much as we can, and adjust how much we have left.
        delta_remaining -= inventory->update(item, delta_per_inventory);
        num_inventories_remaining--;
        if (num_inventories_remaining > 0) {
          delta_per_inventory = delta_remaining / num_inventories_remaining;
        }
      } else {
        next_inventories_to_consider.push_back(inventory);
      }
    }
    // Do this until we don't kick any inventories off the list. Once we're here, all remaining inventories can
    // "fully participate".
  } while (inventories_to_consider.size() != next_inventories_to_consider.size());

  if (num_inventories_remaining == 0) {
    return delta - delta_remaining;
  }

  // Update in reverse order. Because of the direction of rounding, this means that the earlier inventories will get
  // more of the delta (if it's not evenly split).
  for (int i = inventories_to_consider.size() - 1; i >= 0; i--) {
    Inventory* inventory = inventories_to_consider[i];
    InventoryDelta inventory_delta = delta_remaining / (i + 1);
    InventoryDelta actual_delta = inventory->update(item, inventory_delta);
    assert(actual_delta == inventory_delta && "Expected inventory to absorb all of the delta");
    delta_remaining -= actual_delta;
  }
  assert(delta_remaining == 0 && "Expected all of the delta to be consumed");

  return delta - delta_remaining;
}

// Static method for transferring resources between inventories
InventoryDelta HasInventory::transfer_resources(Inventory& source,
                                                Inventory& target,
                                                InventoryItem item,
                                                InventoryDelta delta,
                                                bool destroy_untransferred_resources) {
  // We want to only transfer positive deltas. If you want to transfer negative, switch source and target.
  if (delta <= 0) {
    return 0;
  }

  // Figure out how many resources the source can give
  InventoryQuantity source_available = source.amount(item);
  InventoryDelta max_source_can_give = std::min(static_cast<InventoryDelta>(source_available), delta);

  // Figure out how many resources the target can receive
  InventoryQuantity target_free_space = target.free_space(item);
  InventoryDelta max_target_can_receive = static_cast<InventoryDelta>(target_free_space);

  // Calculate the actual transfer amount
  InventoryDelta transfer_amount = std::min(max_source_can_give, max_target_can_receive);
  InventoryDelta source_loss = destroy_untransferred_resources ? max_source_can_give : transfer_amount;

  // Remove resources from source
  [[maybe_unused]] InventoryDelta actually_removed = source.update(item, -source_loss);
  assert(actually_removed == -source_loss && "Expected source to lose the amount of resources it claimed to lose");

  // Add resources to target
  [[maybe_unused]] InventoryDelta actually_added = target.update(item, transfer_amount);
  assert(actually_added == transfer_amount &&
         "Expected target to receive the amount of resources it claimed to receive");

  return transfer_amount;
}

// Whether the inventory is accessible to an agent.
bool HasInventory::inventory_is_accessible() {
  return true;
}
