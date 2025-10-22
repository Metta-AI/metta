#include "objects/inventory.hpp"

#include <algorithm>
#include <cassert>
#include <limits>
#include <set>
#include <unordered_map>
#include <vector>

// Constructor implementation
Inventory::Inventory(const InventoryConfig& cfg) : _inventory(), _limits() {
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
InventoryDelta Inventory::update(InventoryItem item, InventoryDelta attempted_delta) {
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
  InventoryQuantity total_limit = limit->limit;

  return total_limit - used;
}

// Get method implementation
std::unordered_map<InventoryItem, InventoryQuantity> Inventory::get() const {
  return this->_inventory;
}
