#include "objects/inventory.hpp"

#include <algorithm>
#include <cassert>
#include <limits>
#include <set>
#include <unordered_map>
#include <vector>

#include "objects/has_inventory.hpp"

// Constructor implementation
Inventory::Inventory(const InventoryConfig& cfg,
                     HasInventory* owner,
                     const std::vector<std::string>* resource_names,
                     const std::unordered_map<std::string, ObservationType>* feature_ids)
    : _inventory(), _limits(), _owner(owner) {
  for (const auto& limit_def : cfg.limit_defs) {
    SharedInventoryLimit* limit = new SharedInventoryLimit();
    limit->amount = 0;
    limit->base_limit = limit_def.base_limit;
    limit->modifiers = limit_def.modifiers;
    for (const auto& resource : limit_def.resources) {
      this->_limits[resource] = limit;
    }
  }
}

// Destructor implementation
Inventory::~Inventory() {
  std::set<SharedInventoryLimit*> limits;
  for (const auto& [item, limit] : this->_limits) {
    limits.insert(limit);
  }
  for (const auto& limit : limits) {
    delete limit;
  }
}

// Update method implementation
InventoryDelta Inventory::update(InventoryItem item, InventoryDelta attempted_delta, bool ignore_limits) {
  InventoryQuantity initial_amount = this->_inventory[item];
  int new_amount = static_cast<int>(initial_amount + attempted_delta);

  constexpr InventoryQuantity min = std::numeric_limits<InventoryQuantity>::min();
  InventoryQuantity max = std::numeric_limits<InventoryQuantity>::max();
  SharedInventoryLimit* limit = nullptr;

  if (!ignore_limits && this->_limits.count(item) > 0) {
    limit = this->_limits.at(item);
    // Get effective limit (base + modifiers)
    InventoryQuantity effective = limit->effective_limit(this->_inventory);
    // The max is the total limit, minus whatever's used. But don't count this specific
    // resource.
    int used_by_others = limit->amount - initial_amount;
    if (used_by_others < 0) used_by_others = 0;  // Safety against underflow if something went wrong
    int max_int = static_cast<int>(effective) - used_by_others;
    if (max_int < 0) max_int = 0;
    max = static_cast<InventoryQuantity>(max_int);
  }

  InventoryQuantity clamped_amount = static_cast<InventoryQuantity>(std::clamp<int>(new_amount, min, max));

  if (clamped_amount == 0) {
    this->_inventory.erase(item);
  } else {
    this->_inventory[item] = clamped_amount;
  }

  // Update limit tracking even when ignoring limits (so it reflects actual inventory state)
  if (this->_limits.count(item) > 0) {
    this->_limits.at(item)->amount += clamped_amount - initial_amount;
  }

  InventoryDelta clamped_delta = clamped_amount - initial_amount;

  // Notify owner if inventory actually changed
  if (_owner && clamped_delta != 0) {
    _owner->on_inventory_change(item, clamped_delta);
  }

  // If we removed a modifier item, enforce limits on all resources
  // (their effective limits may have decreased)
  if (clamped_delta < 0 && is_modifier(item)) {
    enforce_all_limits();
  }

  return clamped_delta;
}

// Amount method implementation
InventoryQuantity Inventory::amount(InventoryItem item) const {
  if (this->_inventory.count(item) == 0) {
    return 0;
  }
  return this->_inventory.at(item);
}

// Free space method implementation
InventoryQuantity Inventory::free_space(InventoryItem item) const {
  if (this->_limits.count(item) == 0) {
    return std::numeric_limits<InventoryQuantity>::max() - this->amount(item);
  }

  SharedInventoryLimit* limit = this->_limits.at(item);

  InventoryQuantity used = limit->amount;
  InventoryQuantity effective = limit->effective_limit(this->_inventory);

  // Prevent underflow when used exceeds limit (can happen with dynamic modifiers)
  return effective > used ? effective - used : 0;
}

// Get method implementation
std::unordered_map<InventoryItem, InventoryQuantity> Inventory::get() const {
  return this->_inventory;
}

// Check if an item is a modifier for any limit
bool Inventory::is_modifier(InventoryItem item) const {
  for (const auto& [_, limit] : this->_limits) {
    if (limit->modifiers.count(item) > 0) {
      return true;
    }
  }
  return false;
}

// Enforce all limits - drop excess items when limits decrease
void Inventory::enforce_all_limits() {
  // Collect unique limits (multiple resources can share a limit)
  std::set<SharedInventoryLimit*> unique_limits;
  for (const auto& [_, limit] : this->_limits) {
    unique_limits.insert(limit);
  }

  // For each limit, check if we need to drop excess
  for (SharedInventoryLimit* limit : unique_limits) {
    InventoryDelta excess = limit->amount - limit->effective_limit(this->_inventory);
    if (excess <= 0) {
      continue;  // No excess
    }

    // Find resources belonging to this limit and drop excess
    for (auto& [item, item_limit] : this->_limits) {
      if (item_limit != limit) continue;

      InventoryQuantity current = this->amount(item);

      // Drop up to 'excess' from this item
      InventoryDelta to_drop = std::min(static_cast<InventoryDelta>(current), excess);
      if (to_drop > 0) {
        // update may recurse, if we drop something that impacts inventory limits.
        this->update(item, -to_drop);
        // Because of the potential recursion, calculate the excess again from scratch.
        excess = limit->amount - limit->effective_limit(this->_inventory);
      }

      if (excess <= 0) break;
    }
  }
}
