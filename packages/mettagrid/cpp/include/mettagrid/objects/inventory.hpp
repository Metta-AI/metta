#ifndef PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_INVENTORY_HPP_
#define PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_INVENTORY_HPP_

#include <algorithm>
#include <string>
#include <unordered_map>

#include "core/types.hpp"
#include "objects/constants.hpp"
#include "objects/inventory_config.hpp"
struct SharedInventoryLimit {
  InventoryQuantity limit;
  // How much do we have of whatever-this-limit-applies-to
  InventoryQuantity amount;
};

class Inventory {
private:
  std::unordered_map<InventoryItem, InventoryQuantity> _inventory;
  std::unordered_map<InventoryItem, SharedInventoryLimit*> _limits;

public:
  // Splits the delta between the inventories. Returns the amount of delta successfully consumed.
  static InventoryDelta shared_update(std::vector<Inventory*> inventories, InventoryItem item, InventoryDelta delta) {
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
    // One things we specifically aim for is that the earlier inventories get more of the update, if it can't be
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

  explicit Inventory(const InventoryConfig& cfg) : _inventory(), _limits() {
    for (const auto& limit_pair : cfg.limits) {
      const auto& resources = limit_pair.first;
      const auto& limit_value = limit_pair.second;
      SharedInventoryLimit* limit = new SharedInventoryLimit();
      limit->amount = 0;
      limit->limit = limit_value;
      for (const auto& resource : resources) {
        this->_limits[resource] = limit;
      }
    }
  }

  ~Inventory() {
    std::set<SharedInventoryLimit*> limits;
    for (const auto& [item, limit] : this->_limits) {
      limits.insert(limit);
    }
    for (const auto& limit : limits) {
      delete limit;
    }
  }

  InventoryDelta update(InventoryItem item, InventoryDelta attempted_delta) {
    InventoryQuantity initial_amount = this->_inventory[item];
    int new_amount = static_cast<int>(initial_amount + attempted_delta);

    constexpr InventoryQuantity min = std::numeric_limits<InventoryQuantity>::min();
    InventoryQuantity max = std::numeric_limits<InventoryQuantity>::max();
    SharedInventoryLimit* limit = nullptr;
    if (this->_limits.count(item) > 0) {
      limit = this->_limits.at(item);
      // The max is the total limit, minus whatever's used. But don't count this specific
      // resource.
      max = limit->limit - (limit->amount - initial_amount);
    }

    InventoryQuantity clamped_amount = static_cast<InventoryQuantity>(std::clamp<int>(new_amount, min, max));

    if (clamped_amount == 0) {
      this->_inventory.erase(item);
    } else {
      this->_inventory[item] = clamped_amount;
    }

    if (limit) {
      limit->amount += clamped_amount - initial_amount;
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

  InventoryQuantity free_space(InventoryItem item) const {
    if (this->_limits.count(item) == 0) {
      return std::numeric_limits<InventoryQuantity>::max() - this->amount(item);
    }

    SharedInventoryLimit* limit = this->_limits.at(item);

    InventoryQuantity used = limit->amount;
    InventoryQuantity total_limit = limit->limit;

    return total_limit - used;
  }

  std::unordered_map<InventoryItem, InventoryQuantity> get() const {
    return this->_inventory;
  }
};

#endif  // PACKAGES_METTAGRID_CPP_INCLUDE_METTAGRID_OBJECTS_INVENTORY_HPP_
